"""Upgraded 2D MALIBO Optimizers for Systematic 2x2 Ablation Study:
1. TR-MALIBO      (use_trust_region=True,  use_llm=False): Trust-Region MALIBO (emulating BO -> TuRBO-1)
2. LLM-MALIBO     (use_trust_region=False, use_llm=True):  Global MALIBO + LLM directional offset injection
3. LLM-TR-MALIBO  (use_trust_region=True,  use_llm=True):  Trust-Region MALIBO + LLM meta-control & offset injection

Strict parity with existing 2D baselines (Vanilla-TuRBO, Paper-Exact MALIBO, LLM-Meta-TuRBO):
- Exact same 10 seeds, NFE=300, n_init=15 (NACA0012 + 14 symmetric ±0.03 perturbations -> Step-15 = 72.47 ± 2.24);
- Exact same pretrained 250-task 2D MALIBO ResFFN checkpoint (zero test leakage);
- Exact same TuRBO-1 trust-region state machine (L_init=0.8, L_min=0.5**7, L_max=1.6, success_tol=3, failure_tol=4);
- Exact same stagnation-adaptive LLM protocol (Step 16 post_init + stag >= 10, interval >= 18, step <= 180)
  and 4 orthogonal smooth CST deformation modes + 30% directional candidate injection (15 steps, step <= 195).
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.case_2d.airfoil_utils import theta_to_coordinates
from src.case_2d.base_optimizer import BaseAirfoilOptimizer
from src.case_2d.evaluator import AerodynamicEvaluator
from src.case_2d.llm_agent import LLMMALIBO2DAgent
from src.case_2d.malibo_core import (
    LaplaceTaskAdapter2D,
    MALIBO2DConfig,
    MALIBOMetaModel2D,
    MALIBOResidualBooster2D,
)


def convert_directional_strategy(strat: Optional[Dict[str, Any]]) -> np.ndarray:
    """Convert 4 symmetric orthogonal aerodynamic modes into an 18D Kulfan CST shift vector."""
    delta = np.zeros(18, dtype=np.float64)
    if not strat or not isinstance(strat, dict):
        return delta

    s_camber = float(strat.get("camber_shift", 0.0))
    s_aft = float(strat.get("aft_camber_shift", 0.0))
    s_thick = float(strat.get("thickness_shift", 0.0))
    s_le = float(strat.get("le_radius_shift", 0.0))

    u = (np.arange(8, dtype=np.float64) + 0.5) / 8.0
    bell = np.sin(np.pi * u)
    aft_bell = (u ** 2) * np.sin(np.pi * u)
    aft_bell = aft_bell / max(float(np.max(aft_bell)), 1e-8)

    camber_profile = s_camber * bell + s_aft * aft_bell
    thick_profile = 0.5 * s_thick * bell

    delta[0:8] = camber_profile + thick_profile
    delta[8:16] = camber_profile - thick_profile
    delta[16] = s_le
    delta[17] = 0.0
    return delta


class UpgradedMALIBO2DOptimizer(BaseAirfoilOptimizer):
    """Unified 2D upgraded MALIBO optimizer supporting TR-MALIBO, LLM-MALIBO, and LLM-TR-MALIBO."""

    def __init__(
        self,
        meta_model: MALIBOMetaModel2D,
        evaluator: AerodynamicEvaluator,
        use_trust_region: bool = True,
        use_llm: bool = False,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        config: Optional[MALIBO2DConfig] = None,
        max_evals: int = 300,
        seed: int = 42,
        n_init: int = 15,
        length_init: float = 0.8,
        length_min: float = 0.5 ** 7,
        length_max: float = 1.6,
        success_tol: int = 3,
        failure_tol: int = 4,
        llm_agent: Optional[LLMMALIBO2DAgent] = None,
        milestone_steps: Tuple[int, ...] = (16,),
        stag_threshold: int = 10,
        min_intervention_interval: int = 18,
        global_dir_ref_scale: float = 0.4,
        device: Optional[torch.device] = None,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed, is_meta=True)
        self.meta_model = meta_model
        self.use_trust_region = bool(use_trust_region)
        self.use_llm = bool(use_llm)
        self.config = config if config is not None else MALIBO2DConfig()
        self.n_init = int(n_init)
        self.n_candidates = int(self.config.optimization.n_candidates)  # 5,120 (Appendix G)
        self.gamma = float(self.config.optimization.gamma)              # fixed 1/3

        self.device = device if device is not None else next(meta_model.parameters()).device
        self.meta_model.eval().to(self.device)

        self.adapter = LaplaceTaskAdapter2D(
            meta_model=self.meta_model,
            config=self.config.adaptation,
            gamma=self.gamma,
            device=self.device,
        )
        self.residual_booster = MALIBOResidualBooster2D(
            meta_model=self.meta_model,
            config=self.config.booster,
            device=self.device,
            seed=self.seed,
        )

        # Trust-region state machine (aligned with 2D TuRBO-1 / LLM-Meta-TuRBO)
        self.L_min = float(length_min)
        self.L_max = float(length_max)
        self.L_init = float(length_init)
        self.base_L = float(length_init)
        self.scale_L_mult = 1.0
        self.scale_L_steps_left = 0
        self.L = float(length_init)
        self.success_tol = int(success_tol)
        self.failure_tol = int(failure_tol)
        self.n_success = 0
        self.n_failure = 0
        self.prob_perturb = min(20.0 / len(self.x0), 1.0)
        self.temperature = 1.0

        # LLM stagnation-adaptive directional injection state
        self.llm_agent = llm_agent
        self.milestone_steps = set(milestone_steps)
        self.stag_threshold = int(stag_threshold)
        self.min_intervention_interval = int(min_intervention_interval)
        self.last_intervention_step = 0
        self.guide_direction_unit: Optional[np.ndarray] = None
        self.guide_direction_steps_left: int = 0
        self.global_dir_ref_scale = float(global_dir_ref_scale)
        self.intervention_history: List[Dict[str, Any]] = []

    @property
    def variant_name(self) -> str:
        if self.use_trust_region and self.use_llm:
            return "LLM-TR-MALIBO"
        if self.use_trust_region:
            return "TR-MALIBO"
        if self.use_llm:
            return "LLM-MALIBO"
        return "MALIBO"

    def get_state(self) -> np.ndarray:
        return self.get_base_state()

    def apply_action(self, action: np.ndarray):
        pass

    def _adjust_trust_region(self, improved: bool) -> None:
        if not self.use_trust_region:
            return

        if improved:
            self.n_success += 1
            self.n_failure = 0
        else:
            self.n_success = 0
            self.n_failure += 1

        if self.n_success >= self.success_tol:
            self.base_L = min(self.base_L * 2.0, self.L_max)
            self.n_success = 0
        elif self.n_failure >= self.failure_tol:
            self.base_L = self.base_L / 2.0
            if self.base_L < self.L_min * 1.5:
                self.base_L = self.L_init
                self.scale_L_mult = 1.0
                self.scale_L_steps_left = 0
            self.n_failure = 0

        if self.scale_L_steps_left > 0:
            self.scale_L_steps_left -= 1
            if self.scale_L_steps_left == 0:
                self.scale_L_mult = 1.0

        self.L = float(np.clip(self.base_L * self.scale_L_mult, self.L_min, self.L_max))

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        d = len(self.x0)
        dim_range = self.upper_bounds - self.lower_bounds

        def to_unit(x: np.ndarray) -> np.ndarray:
            return np.clip((x - self.lower_bounds) / (dim_range + 1e-12), 0.0, 1.0)

        def from_unit(u: np.ndarray) -> np.ndarray:
            return np.clip(self.lower_bounds + u * dim_range, self.lower_bounds, self.upper_bounds)

        X_unit: List[np.ndarray] = []
        Y_min: List[float] = []

        # 1. Exact bit-for-bit aligned n_init = 15 initialization
        score0 = self.evaluate(self.x0)
        self.initial_score = score0
        X_unit.append(to_unit(self.x0))
        Y_min.append(-float(score0))

        for _ in range(1, self.n_init):
            if self.eval_count >= self.max_evals:
                break
            noise = self.rng.uniform(-0.03, 0.03, size=d)
            x_cand = np.clip(self.x0 + noise, self.lower_bounds, self.upper_bounds)
            score = self.evaluate(x_cand)
            X_unit.append(to_unit(x_cand))
            Y_min.append(-float(score))

        # 2. Main optimization loop
        while self.eval_count < self.max_evals:
            score_before = self.best_score
            step = self.eval_count + 1

            # 2a. Optional stagnation-adaptive LLM intervention
            if self.use_llm and self.llm_agent is not None:
                is_milestone = (step in self.milestone_steps)
                is_stagnation = (
                    self.stagnation_counter >= self.stag_threshold
                    and (step - self.last_intervention_step) >= self.min_intervention_interval
                )
                need_intervention = (step <= 180) and (is_milestone or is_stagnation)

                if need_intervention:
                    self.last_intervention_step = step
                    trigger_reason = (
                        "post_init" if (is_milestone and not is_stagnation and step == 16) else "stagnation"
                    )
                    det = self.evaluator.evaluate_detailed(self.best_x)
                    state_dict = {
                        "airfoil_name": "NACA0012",
                        "reynolds": self.evaluator.Re,
                        "alpha": self.evaluator.alpha,
                        "step": step,
                        "trigger_reason": trigger_reason,
                        "max_evals": self.max_evals,
                        "budget_left": self.max_evals - self.eval_count,
                        "dim": d,
                        "init_ld": self.initial_score,
                        "best_ld": self.best_score,
                        "best_cl": det.get("cl", 0.0),
                        "best_cd": det.get("cd", 0.0),
                        "best_cm": det.get("cm", 0.0),
                        "best_t_max": det.get("t_max", 0.0),
                        "stagnation_count": self.stagnation_counter,
                        "tr_length": self.L,
                        "success_counter": self.n_success,
                        "success_tolerance": self.success_tol,
                    }

                    action = self.llm_agent.decide_action(state_dict)
                    inter_item: Dict[str, Any] = {
                        "step": step,
                        "trigger_reason": trigger_reason,
                        "action": action,
                    }
                    if hasattr(self.llm_agent, "interaction_log") and len(self.llm_agent.interaction_log) > 0:
                        last_log = self.llm_agent.interaction_log[-1]
                        inter_item["user_prompt"] = last_log.get("user_prompt")
                        inter_item["raw_response"] = last_log.get("raw_response")
                    self.intervention_history.append(inter_item)

                    # Apply LLM meta-parameters
                    self.temperature = float(action["exploration_weight"])
                    if self.use_trust_region:
                        self.scale_L_mult = float(action["length_scale_factor"])
                        self.scale_L_steps_left = 12
                        self.L = float(np.clip(self.base_L * self.scale_L_mult, self.L_min, self.L_max))
                        self.prob_perturb = float(action["prob_perturb"])

                    # Analytical zero-NFE geometry feasibility backtracking on macro shift
                    strat = action.get("directional_strategy")
                    delta_dir = convert_directional_strategy(strat)
                    cand_to_eval = None
                    if np.linalg.norm(delta_dir) > 1e-8:
                        for s_fac in (1.0, 0.5, 0.25):
                            trial_x = np.clip(self.best_x + s_fac * delta_dir, self.lower_bounds, self.upper_bounds)
                            try:
                                _, k_af = self.evaluator._theta_to_kulfan(trial_x)
                                is_fatal, _, metrics = self.evaluator.validate_geometry(k_af)
                                is_geom_ok, total_penalty, _, _ = self.evaluator.calculate_penalties(metrics)
                                if not is_fatal and is_geom_ok and total_penalty == 0.0:
                                    cand_to_eval = trial_x
                                    break
                            except Exception:
                                pass

                    if cand_to_eval is not None:
                        u_best = to_unit(self.best_x)
                        u_cand = to_unit(cand_to_eval)
                        diff_u = u_cand - u_best
                        norm_u = float(np.linalg.norm(diff_u))
                        if norm_u > 1e-4:
                            self.guide_direction_unit = diff_u / norm_u
                            self.guide_direction_steps_left = 15

                        cand_score = self.evaluate(cand_to_eval)
                        X_unit.append(u_cand)
                        Y_min.append(-float(cand_score))

                        improved = (cand_score > score_before + 1e-3)
                        self._adjust_trust_region(improved)

                        if self.eval_count >= self.max_evals:
                            break
                        continue

            # 2b. MALIBO Laplace adaptation + 70/30 early-stopped residual GBDT
            X_arr = np.array(X_unit, dtype=np.float64)
            y_clean = np.minimum(np.array(Y_min, dtype=np.float64), 10.0)
            y_mean = float(np.mean(y_clean))
            y_std_val = max(float(np.std(y_clean)), 1e-4)
            y_norm = (y_clean - y_mean) / y_std_val
            tau_std = float(np.quantile(y_norm, self.gamma))

            t_X = torch.as_tensor(X_arr, dtype=torch.float32, device=self.device)
            t_y = torch.as_tensor(y_norm, dtype=torch.float32, device=self.device)

            z_sample, _ = self.adapter.adapt_and_sample(
                t_X, t_y, tau_std=tau_std, rng=self.rng, temperature=self.temperature
            )
            self.residual_booster.fit_with_early_stopping(
                X=X_arr,
                y_std=y_norm,
                tau_std=tau_std,
                z_sample=z_sample,
                step_seed=int(self.seed * 1009 + step),
            )

            # 2c. Candidate generation: Trust-Region vs Global + optional 30% LLM directional injection
            x_best_unit = to_unit(self.best_x)
            n_cand = self.n_candidates

            if self.use_trust_region:
                tr_lb = np.clip(x_best_unit - self.L / 2.0, 0.0, 1.0)
                tr_ub = np.clip(x_best_unit + self.L / 2.0, 0.0, 1.0)
                cand_samples = self.rng.uniform(tr_lb, tr_ub, size=(n_cand, d))
                mask = self.rng.rand(n_cand, d) > self.prob_perturb
                mask[self.rng.rand(n_cand) < 0.1, :] = False
                cand_samples[mask] = np.broadcast_to(x_best_unit, (n_cand, d))[mask]
                clip_lb, clip_ub = tr_lb, tr_ub
                dir_scale = self.L
            else:
                cand_samples = self.rng.uniform(0.0, 1.0, size=(n_cand, d))
                clip_lb, clip_ub = 0.0, 1.0
                dir_scale = self.global_dir_ref_scale

            if (
                self.use_llm
                and self.guide_direction_unit is not None
                and self.guide_direction_steps_left > 0
                and step <= 195
            ):
                n_dir = min(1536, n_cand // 3)
                drift = self.rng.uniform(0.05 * dir_scale, 0.65 * dir_scale, size=(n_dir, 1))
                dir_cands = (
                    x_best_unit
                    + drift * self.guide_direction_unit
                    + self.rng.normal(0.0, 0.03 * dir_scale, size=(n_dir, d))
                )
                cand_samples[:n_dir] = np.clip(dir_cands, clip_lb, clip_ub)
                self.guide_direction_steps_left -= 1

            # 2d. Maximize MALIBO residual-boosted acquisition utility
            utilities = self.residual_booster.predict_utility(
                X_candidates=cand_samples,
                z_sample=z_sample,
                use_raw_margin=True,
            )
            next_x_unit = cand_samples[int(np.argmax(utilities))]
            next_x = from_unit(next_x_unit)

            new_score = self.evaluate(next_x)
            X_unit.append(next_x_unit)
            Y_min.append(-float(new_score))

            improved = (new_score > score_before + 1e-3)
            self._adjust_trust_region(improved)

        while len(self.history_best) < self.max_evals:
            self.history_best.append(self.best_score)

        elapsed = time.time() - t0
        det = self.evaluator.evaluate_detailed(self.best_x)
        coords = theta_to_coordinates(self.best_x)

        out: Dict[str, Any] = {
            "method": self.variant_name,
            "seed": self.seed,
            "best_score": float(self.best_score),
            "history": [float(v) for v in self.history_best[: self.max_evals]],
            "best_x": [float(v) for v in self.best_x],
            "coordinates": {k: v.tolist() for k, v in coords.items()},
            "detailed": {
                "cl": float(det.get("cl", 0.0)),
                "cd": float(det.get("cd", 0.0)),
                "cm": float(det.get("cm", 0.0)),
                "t_max": float(det.get("t_max", 0.0)),
                "x_t_max": float(det.get("x_t_max", 0.0)),
                "area": float(det.get("area", 0.0)),
                "is_feasible": bool(det.get("is_feasible", True)),
            },
            "elapsed_seconds": float(elapsed),
        }
        if self.use_llm:
            out["interventions"] = self.intervention_history
        return out
