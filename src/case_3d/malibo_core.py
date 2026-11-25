"""Self-contained 3D MALIBO neural meta-model, Laplace adapter, and residual GBDT booster.

Strictly preserves the exact state-dict architecture of `weights/malibo_3d_checkpoint.pt`
(Pan et al., ICML 2024: 4-layer 64-unit ELU ResFFN -> d=50, MeanHead, L-BFGS Laplace MAP/Hessian,
and 70/30 early-stopped LFBO residual GBDT), adding only an optional Laplace Thompson-sampling
temperature multiplier `temperature` (default 1.0 = exact Paper-Exact Eq. 9).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


@dataclass
class ResFFNConfig:
    in_dim: int = 63
    hidden_dim: int = 64
    num_hidden_layers: int = 4
    feature_dim: int = 50
    activation: str = "elu"


@dataclass
class AdaptationConfig:
    lr: float = 1.0
    max_iter: int = 20
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: str = "strong_wolfe"


@dataclass
class BoosterConfig:
    n_estimators: int = 100
    learning_rate: float = 0.1
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_depth: int = 3
    val_ratio: float = 0.30


@dataclass
class OptimizationConfig:
    n_candidates: int = 5120
    budget: int = 300
    seed: int = 42
    n_init: int = 15
    gamma: float = 1.0 / 3.0


@dataclass
class MALIBO3DConfig:
    resffn: ResFFNConfig = field(default_factory=ResFFNConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    booster: BoosterConfig = field(default_factory=BoosterConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + residual)


class ResFFN(nn.Module):
    def __init__(self, config: Optional[ResFFNConfig] = None):
        super().__init__()
        if config is None:
            config = ResFFNConfig()
        self.in_dim = config.in_dim
        self.hidden_dim = config.hidden_dim
        self.feature_dim = config.feature_dim
        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.act = nn.ELU()
        self.res_blocks = nn.ModuleList(
            [ResBlock(self.hidden_dim) for _ in range(config.num_hidden_layers)]
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_proj(x))
        for block in self.res_blocks:
            h = block(h)
        return self.output_proj(h)


class MeanHead(nn.Module):
    def __init__(self, feature_dim: int = 50, hidden_dim: int = 50):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(phi)))


class MALIBOMetaModel3D(nn.Module):
    """Exact state-dict compatible 3D MALIBO meta-model (uses `self.phi_net` attribute)."""

    def __init__(self, num_tasks: int, config: Optional[ResFFNConfig] = None):
        super().__init__()
        if config is None:
            config = ResFFNConfig()
        self.num_tasks = num_tasks
        self.feature_dim = config.feature_dim
        self.phi_net = ResFFN(config)
        self.mean_head = MeanHead(feature_dim=self.feature_dim, hidden_dim=self.feature_dim)
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, self.feature_dim) * 0.1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi_net(x)

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.extract_features(x)
        return self.mean_head(phi)

    def predict_proba(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.forward(x, z=z)
        return torch.sigmoid(logits)

    def forward(
        self,
        x: torch.Tensor,
        task_idx: Optional[Union[int, torch.Tensor]] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        phi = self.extract_features(x)
        m = self.mean_head(phi).squeeze(-1)
        if z is not None:
            interaction = torch.matmul(phi, z) if z.dim() == 1 else torch.sum(phi * z, dim=-1)
            return m + interaction
        if task_idx is not None:
            z_t = self.task_embeddings[task_idx]
            interaction = torch.matmul(phi, z_t) if isinstance(task_idx, int) else torch.sum(phi * z_t, dim=-1)
            return m + interaction
        return m


def load_3d_malibo_model(checkpoint_path: str, device: torch.device) -> MALIBOMetaModel3D:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    res_cfg = ResFFNConfig(
        in_dim=int(ckpt.get("in_dim", 63)),
        hidden_dim=int(ckpt.get("hidden_dim", 64)),
        num_hidden_layers=int(ckpt.get("num_hidden_layers", 4)),
        feature_dim=int(ckpt.get("feature_dim", 50)),
    )
    model = MALIBOMetaModel3D(num_tasks=int(ckpt["num_tasks"]), config=res_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval().to(device)
    return model


class LaplaceTaskAdapter3D:
    """Online Laplace MAP + exact negative Hessian precision + Thompson sampling."""

    def __init__(
        self,
        meta_model: MALIBOMetaModel3D,
        config: Optional[AdaptationConfig] = None,
        gamma: float = 1.0 / 3.0,
        device: Optional[torch.device] = None,
    ):
        self.meta_model = meta_model
        self.config = config if config is not None else AdaptationConfig()
        self.gamma = float(gamma)
        self.device = device if device is not None else next(meta_model.parameters()).device
        self.feature_dim = meta_model.feature_dim
        self.meta_model.eval()
        for p in self.meta_model.parameters():
            p.requires_grad = False

    def adapt_and_sample(
        self,
        x_obs: torch.Tensor,
        y_obs_std: torch.Tensor,
        tau_std: float,
        rng: Optional[np.random.RandomState] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.feature_dim
        if len(x_obs) == 0:
            eps_np = rng.randn(d) if rng is not None else np.random.randn(d)
            z_hat = torch.as_tensor(eps_np * float(temperature), dtype=torch.float32, device=self.device)
            z_map = torch.zeros(d, dtype=torch.float32, device=self.device)
            return z_hat, z_map, torch.eye(d, dtype=torch.float32, device=self.device)

        x_obs = x_obs.to(self.device)
        y_obs_std = y_obs_std.to(self.device)
        w = torch.clamp(float(tau_std) - y_obs_std, min=0.0)

        with torch.no_grad():
            phi = self.meta_model.extract_features(x_obs)
            m = self.meta_model.mean_head(phi).squeeze(-1)

        z = torch.zeros(d, dtype=torch.float32, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS(
            [z],
            lr=self.config.lr,
            max_iter=self.config.max_iter,
            tolerance_grad=self.config.tolerance_grad,
            tolerance_change=self.config.tolerance_change,
            history_size=self.config.history_size,
            line_search_fn=self.config.line_search_fn,
        )

        def closure():
            opt.zero_grad()
            a = m + torch.matmul(phi, z)
            nll = torch.sum(w * F.softplus(-a) + F.softplus(a))
            prior = 0.5 * torch.sum(z ** 2)
            loss = prior + nll
            loss.backward()
            return loss

        opt.step(closure)
        z_map = z.detach()

        with torch.no_grad():
            a_map = m + torch.matmul(phi, z_map)
            k_hat = torch.sigmoid(a_map)
            weights_laplace = (w + 1.0) * k_hat * (1.0 - k_hat)
            weighted_phi = phi * weights_laplace.unsqueeze(-1)
            prec = torch.eye(d, dtype=torch.float32, device=self.device) + (weighted_phi.T @ phi)

            prec_64 = prec.to(torch.float64)
            prec_64 = 0.5 * (prec_64 + prec_64.T)
            diag_scale = torch.clamp(torch.diag(prec_64), min=1.0)
            jitter = torch.diag(diag_scale * 1e-6)

            try:
                L_64 = torch.linalg.cholesky(prec_64 + jitter)
            except torch.linalg.LinAlgError:
                eigvals, eigvecs = torch.linalg.eigh(prec_64)
                eigvals = torch.clamp(eigvals, min=1e-5)
                prec_psd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                L_64 = torch.linalg.cholesky(prec_psd)

            L_prec = L_64.to(torch.float32)

            eps_np = rng.randn(d, 1) if rng is not None else np.random.randn(d, 1)
            eps = torch.as_tensor(eps_np, dtype=torch.float32, device=self.device)
            delta = torch.linalg.solve_triangular(L_prec.T, eps, upper=True).squeeze(-1)
            z_hat = z_map + float(temperature) * delta

        return z_hat, z_map, L_prec


class MALIBOInitialEstimator3D(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        meta_model: MALIBOMetaModel3D,
        z_sample: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        self.meta_model = meta_model
        self.z_sample = z_sample.detach()
        self.device = device if device is not None else next(meta_model.parameters()).device
        self.classes_ = np.array([0, 1], dtype=np.int32)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None):
        self.classes_ = np.array([0, 1], dtype=np.int32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tX = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p1 = self.meta_model.predict_proba(tX, z=self.z_sample).detach().cpu().numpy().reshape(-1)
        p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - p1, p1])


def _build_lfbo_classification_data(
    X: np.ndarray,
    y_std: np.ndarray,
    tau_std: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    w_all = np.maximum(float(tau_std) - y_std, 0.0)
    pos_mask = w_all > 1e-8
    if not np.any(pos_mask):
        return None
    X_pos = X[pos_mask]
    y_pos = np.ones(len(X_pos), dtype=np.int32)
    w_pos = w_all[pos_mask]
    X_neg = X
    y_neg = np.zeros(len(X_neg), dtype=np.int32)
    w_neg = np.ones(len(X_neg), dtype=np.float64)
    return np.vstack([X_pos, X_neg]), np.concatenate([y_pos, y_neg]), np.concatenate([w_pos, w_neg])


class MALIBOResidualBooster3D:
    """70/30 early-stopped LFBO residual GBDT classifier (Pan et al., ICML 2024 Eq. 10 & Appendix G)."""

    def __init__(
        self,
        meta_model: MALIBOMetaModel3D,
        config: Optional[BoosterConfig] = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.meta_model = meta_model
        self.config = config if config is not None else BoosterConfig()
        self.device = device if device is not None else next(meta_model.parameters()).device
        self.seed = int(seed)
        self.booster: Optional[GradientBoostingClassifier] = None
        self.is_fitted = False

    def fit_with_early_stopping(
        self,
        x_obs: np.ndarray,
        y_std: np.ndarray,
        tau_std: float,
        z_sample: torch.Tensor,
        step_seed: Optional[int] = None,
    ) -> bool:
        N = len(x_obs)
        if N < 10:
            self.is_fitted = False
            self.booster = None
            return False

        full_lfbo = _build_lfbo_classification_data(x_obs, y_std, tau_std)
        if full_lfbo is None:
            self.is_fitted = False
            self.booster = None
            return False

        init_est = MALIBOInitialEstimator3D(
            meta_model=self.meta_model,
            z_sample=z_sample,
            device=self.device,
        )
        split_seed = self.seed if step_seed is None else int(step_seed)

        idx_all = np.arange(N)
        k_orig = (y_std <= tau_std).astype(np.int32)
        try:
            idx_tr, idx_val = train_test_split(
                idx_all,
                test_size=self.config.val_ratio,
                random_state=split_seed,
                stratify=k_orig if np.min(np.bincount(k_orig)) >= 2 else None,
            )
        except ValueError:
            idx_tr, idx_val = train_test_split(
                idx_all,
                test_size=self.config.val_ratio,
                random_state=split_seed,
            )

        tr_lfbo = _build_lfbo_classification_data(x_obs[idx_tr], y_std[idx_tr], tau_std)
        val_lfbo = _build_lfbo_classification_data(x_obs[idx_val], y_std[idx_val], tau_std)
        if tr_lfbo is None or val_lfbo is None:
            self.is_fitted = False
            self.booster = None
            return False

        X_tr, y_tr, w_tr = tr_lfbo
        X_val, y_val, w_val = val_lfbo

        p0_val = init_est.predict_proba(X_val)
        loss_0 = float(log_loss(y_val, p0_val, sample_weight=w_val, labels=[0, 1]))

        pilot = GradientBoostingClassifier(
            init=init_est,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=split_seed,
        )
        pilot.fit(X_tr, y_tr, sample_weight=w_tr)

        val_losses = [loss_0]
        for p_stage in pilot.staged_predict_proba(X_val):
            val_losses.append(float(log_loss(y_val, p_stage, sample_weight=w_val, labels=[0, 1])))

        best_trees = int(np.argmin(val_losses))
        if best_trees == 0:
            self.is_fitted = False
            self.booster = None
            return False

        X_full, y_full, w_full = full_lfbo
        final_clf = GradientBoostingClassifier(
            init=init_est,
            n_estimators=best_trees,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=split_seed,
        )
        final_clf.fit(X_full, y_full, sample_weight=w_full)
        self.booster = final_clf
        self.is_fitted = True
        return True

    def predict_utility(
        self,
        x_cand: np.ndarray,
        z_sample: torch.Tensor,
        use_raw_margin: bool = True,
    ) -> np.ndarray:
        if self.is_fitted and self.booster is not None:
            if use_raw_margin:
                return np.asarray(self.booster.decision_function(x_cand), dtype=np.float64).reshape(-1)
            return np.asarray(self.booster.predict_proba(x_cand)[:, 1], dtype=np.float64).reshape(-1)

        with torch.no_grad():
            tX = torch.as_tensor(x_cand, dtype=torch.float32, device=self.device)
            return self.meta_model(tX, z=z_sample).detach().cpu().numpy().reshape(-1)
