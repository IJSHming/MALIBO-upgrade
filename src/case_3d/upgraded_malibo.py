"""CoLT-MALIBO (Constrained LLM-guided Trust-Region MALIBO) 3D Blade Optimizer & Ablation Suite:
- CoLT-MALIBO : Full method (Obj-TR + Con-TR + LLM + feasibility-aware TR contraction + gamma annealing)
- w/o LLM     : Removes LLM directional offset injection & meta-control
- w/o Obj-TR  : Removes TuRBO-style objective trust-region box (global + Con-TR + LLM)
- w/o Con-TR  : Removes SCBO-style Matérn-5/2 constrained feasibility surrogate (Obj-TR + LLM)

Strict parity with frozen 3D baselines (TuRBO-1, Paper-Exact MALIBO, LLM-Meta-TuRBO):
- Exact same 10 seeds, NFE=300, n_init=15 (Phase VI baseline + 14 symmetric ±0.04 unit perturbations -> 6.1773 ± 0.0280 kW);
- Exact same pretrained 100-task 3D MALIBO ResFFN checkpoint (zero test leakage);
- Exact same TuRBO-1 trust-region state machine (L_init=0.8, L_min=0.5**7, L_max=1.6, success_tol=3, failure_tol=6,
  prob_perturb = 20/63 ~= 0.3175);
- Exact same stagnation-adaptive LLM protocol (Step 16 post_init + stag >= 10, interval >= 18, step <= 180)
  and 12 zone-aligned smooth physical modes + 30% directional candidate injection (15 steps, step <= 195).
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.case_3d.base_optimizer import BaseBladeOptimizer
from src.case_3d.blade_geometry import unpack_blade_params
from src.case_3d.evaluator import WindTurbineBladeEvaluator
from src.case_3d.llm_agent import LLMMALIBO3DAgent
from src.case_3d.malibo_core import (
    LaplaceTaskAdapter3D,
    MALIBO3DConfig,
    MALIBOMetaModel3D,
    MALIBOResidualBooster3D,
)
from src.constrained_acquisition import (
    annealed_lfbo_gamma,
    compute_constrained_lfbo_acquisition,
)


def convert_directional_strategy_to_blade_delta(
    strat: Optional[Dict[str, Any]],
    dim: int = 63,
) -> np.ndarray:
    """Convert 12 zone-aligned aerodynamic macro shifts into a 63D B-Spline blade parameter delta."""
    delta = np.zeros(dim, dtype=np.float64)
    if not strat or not isinstance(strat, dict):
        return delta

    station_configs_3 = [
        (0.85, 0.90, 0.85, 1.0, "camber_shift_root", "thickness_shift_root", "chord_shift_root", "twist_shift_root"),
        (1.00, 1.00, 1.00, 1.0, "camber_shift_mid",  "thickness_shift_mid",  "chord_shift_mid",  "twist_shift_mid"),
        (0.85, 0.90, 0.70, 1.0, "camber_shift_tip",  "thickness_shift_tip",  "chord_shift_tip",  "twist_shift_tip"),
    ]
    bell = np.sin(np.pi * (np.arange(8) + 0.5) / 8.0)

    if dim == 63:
        for k in range(3):
            c_sc, t_sc, _, _, c_key, t_key, _, _ = station_configs_3[k]
            s_camber = float(strat.get(c_key, 0.0)) * c_sc
            s_thick = float(strat.get(t_key, 0.0)) * t_sc

            idx = k * 17
            delta[idx : idx + 8] = (s_camber + 0.5 * s_thick) * bell
            delta[idx + 8 : idx + 16] = (s_camber - 0.5 * s_thick) * bell
            delta[idx + 16] = 0.25 * s_thick

        delta[51:53] = float(strat.get("chord_shift_root", 0.0)) * 0.85
        delta[53:55] = float(strat.get("chord_shift_mid", 0.0)) * 1.0
        delta[55:57] = float(strat.get("chord_shift_tip", 0.0)) * 0.70

        delta[57:59] = float(strat.get("twist_shift_root", 0.0)) * 0.90
        delta[59:61] = float(strat.get("twist_shift_mid", 0.0)) * 1.0
        delta[61:63] = float(strat.get("twist_shift_tip", 0.0)) * 1.0

    return delta


class UpgradedMALIBOBladeOptimizer(BaseBladeOptimizer):
    """Unified 3D upgraded MALIBO optimizer supporting TR-MALIBO, LLM-MALIBO, and LLM-TR-MALIBO."""

    def __init__(
        self,
        meta_model: MALIBOMetaModel3D,
        evaluator: WindTurbineBladeEvaluator,
        use_trust_region: bool = True,
        use_llm: bool = False,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        config: Optional[MALIBO3DConfig] = None,
        max_evals: int = 300,
        seed: int = 42,
        n_init: int = 15,
        length_init: float = 0.8,
        length_min: float = 0.5 ** 7,
        length_max: float = 1.6,
        success_tol: int = 3,
        failure_tol: int = 6,
        llm_agent: Optional[LLMMALIBO3DAgent] = None,
        milestone_steps: Tuple[int, ...] = (16,),
        stag_threshold: int = 10,
        min_intervention_interval: int = 18,
        global_dir_ref_scale: float = 0.4,
        use_feas_surrogate: bool = True,
        use_gamma_annealing: bool = True,
        feas_lengthscale: float = 0.35,
        feas_tau: float = 0.55,
        feas_beta: float = 2.0,
        gamma_end: float = 0.15,
        infeas_failure_weight: int = 2,
        device: Optional[torch.device] = None,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed)
        self.meta_model = meta_model
        self.use_trust_region = bool(use_trust_region)
        self.use_llm = bool(use_llm)
        self.config = config if config is not None else MALIBO3DConfig()
        self.n_init = int(n_init)
        self.n_candidates = int(self.config.optimization.n_candidates)  # 5,120 (Appendix G)
        self.gamma = float(self.config.optimization.gamma)              # fixed 1/3

        self.device = device if device is not None else next(meta_model.parameters()).device
        self.meta_model.eval().to(self.device)

        self.adapter = LaplaceTaskAdapter3D(
            meta_model=self.meta_model,
            config=self.config.adaptation,
            gamma=self.gamma,
            device=self.device,
        )
        self.residual_booster = MALIBOResidualBooster3D(
            meta_model=self.meta_model,
            config=self.config.booster,
            device=self.device,
            seed=self.seed,
        )

        # Trust-region state machine (aligned with 3D TuRBO-1 / LLM-Meta-TuRBO)
        self.L_init = float(length_init)
        self.L_min = float(length_min)
        self.L_max = float(length_max)
        self.base_L = float(length_init)
        self.scale_L_mult = 1.0
        self.scale_L_steps_left = 0
        self.L = float(length_init)
        self.success_tol = int(success_tol)
        self.failure_tol = int(failure_tol)
        self.n_success = 0
        self.n_failure = 0
        self.prob_perturb = min(20.0 / self.dim, 1.0)
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

        # Constrained / OOD-Aware acquisition & feasibility-aware TR settings (unified across 2D & 3D)
        self.use_feas_surrogate = bool(use_feas_surrogate)
        self.use_gamma_annealing = bool(use_gamma_annealing)
        self.feas_lengthscale = float(feas_lengthscale)
        self.feas_tau = float(feas_tau)
        self.feas_beta = float(feas_beta)
        self.gamma_end = float(gamma_end)
        self.infeas_failure_weight = int(infeas_failure_weight)

    @property
    def variant_name(self) -> str:
        if self.use_trust_region and self.use_llm and self.use_feas_surrogate:
            return "CoLT-MALIBO"
        if self.use_trust_region and (not self.use_llm) and self.use_feas_surrogate:
            return "w/o LLM"
        if (not self.use_trust_region) and self.use_llm and self.use_feas_surrogate:
            return "w/o Obj-TR"
        if self.use_trust_region and self.use_llm and (not self.use_feas_surrogate):
            return "w/o Con-TR"
        return "MALIBO"

    def _adjust_trust_region(self, improved: bool, is_feasible: bool = True) -> None:
        if not self.use_trust_region:
            return

        if improved:
            self.n_success += 1
            self.n_failure = 0
        else:
            self.n_success = 0
            fail_inc = self.infeas_failure_weight if (not is_feasible) else 1
            self.n_failure += fail_inc

        if self.n_success >= self.success_tol:
            self.base_L = min(self.base_L * 2.0, self.L_max)
            self.n_success = 0
        elif self.n_failure >= self.failure_tol:
            self.base_L = self.base_L / 2.0
            self.n_failure = 0

        if self.base_L < self.L_min:
            self.base_L = self.L_init
            self.n_success = 0
            self.n_failure = 0
            self.scale_L_mult = 1.0
            self.scale_L_steps_left = 0

        if self.scale_L_steps_left > 0:
            self.scale_L_steps_left -= 1
            if self.scale_L_steps_left == 0:
                self.scale_L_mult = 1.0

        self.L = float(np.clip(self.base_L * self.scale_L_mult, self.L_min, self.L_max))

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        d = self.dim

        X_unit_list: List[np.ndarray] = []
        Y_minimize_list: List[float] = []
        C_feas_list: List[float] = []

        # 1. Exact bit-for-bit aligned n_init = 15 initialization
        score0 = self.evaluate_initial()
        u0 = self.to_unit(self.x0)
        X_unit_list.append(u0)
        Y_minimize_list.append(-float(score0))
        C_feas_list.append(1.0 if (score0 > 0.0) else 0.0)

        for _ in range(1, self.n_init):
            if self.eval_count >= self.max_evals:
                break
            u_cand = np.clip(u0 + self.rng.uniform(-0.04, 0.04, size=d), 0.0, 1.0)
            x_cand = self.from_unit(u_cand)
            score = self.evaluate(x_cand)
            X_unit_list.append(u_cand)
            Y_minimize_list.append(-float(score))
            C_feas_list.append(1.0 if (score > 0.0) else 0.0)

        # 2. Main optimization loop
        while self.eval_count < self.max_evals:
            step = self.eval_count + 1
            score_before = self.best_score

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
                    if hasattr(self.evaluator, "param_model") and hasattr(self.evaluator.param_model, "get_station_info"):
                        stations = self.evaluator.param_model.get_station_info(self.best_x)
                    else:
                        stations = unpack_blade_params(self.best_x)

                    st_metrics_list = det.get("geom_metrics", {}).get("station_metrics", [])
                    state_dict = {
                        "step": step,
                        "trigger_reason": trigger_reason,
                        "max_evals": self.max_evals,
                        "budget_left": self.max_evals - self.eval_count,
                        "wind_speed": self.evaluator.wind_speed,
                        "rpm": self.evaluator.rpm,
                        "omega": self.evaluator.omega,
                        "init_power_kW": self.initial_score,
                        "best_power_kW": self.best_score,
                        "best_torque_kNm": det.get("torque_kNm", 0.0),
                        "best_thrust_kN": det.get("thrust_kN", 0.0),
                        "max_allowable_thrust_kN": self.evaluator.max_allowable_thrust_kN,
                        "best_Cp": det.get("Cp", 0.0),
                        "best_Ct": det.get("Ct", 0.0),
                        "stagnation_counter": self.stagnation_counter,
                        "tr_length": self.L,
                        "success_counter": self.n_success,
                        "success_tolerance": self.success_tol,
                        "station_metrics": [
                            {
                                "radius": s["radius"],
                                "chord": s["chord"],
                                "twist": s["twist"],
                                "max_t_rel": st_metrics_list[k].get("max_t_rel", 0.21) if k < len(st_metrics_list) else 0.21,
                            }
                            for k, s in enumerate(stations)
                        ],
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
                    self.temperature = float(action.get("exploration_weight", 1.0))
                    if self.use_trust_region:
                        self.scale_L_mult = float(action.get("length_scale_factor", 1.0))
                        self.scale_L_steps_left = 12
                        self.L = float(np.clip(self.base_L * self.scale_L_mult, self.L_min, self.L_max))
                        self.prob_perturb = float(action.get("prob_perturb", min(20.0 / d, 1.0)))

                    # Analytical zero-NFE geometry feasibility backtracking on macro shift
                    strat = action.get("directional_strategy")
                    delta_dir = convert_directional_strategy_to_blade_delta(strat, dim=d)
                    cand_to_eval = None
                    if np.linalg.norm(delta_dir) > 1e-8:
                        for s_fac in (1.0, 0.5, 0.25):
                            trial_x = np.clip(self.best_x + s_fac * delta_dir, self.lower_bounds, self.upper_bounds)
                            is_fatal, _, metrics = self.evaluator.validate_geometry(trial_x)
                            is_geom_ok, total_penalty, _ = self.evaluator.calculate_penalties(metrics)
                            if not is_fatal and is_geom_ok and total_penalty == 0.0:
                                cand_to_eval = trial_x
                                break

                    if cand_to_eval is not None:
                        u_best = self.to_unit(self.best_x)
                        u_cand = self.to_unit(cand_to_eval)
                        diff_u = u_cand - u_best
                        norm_u = float(np.linalg.norm(diff_u))
                        if norm_u > 1e-4:
                            self.guide_direction_unit = diff_u / norm_u
                            self.guide_direction_steps_left = 15

                        cand_score = self.evaluate(cand_to_eval)
                        cand_feas = (cand_score > 0.0)
                        X_unit_list.append(u_cand)
                        Y_minimize_list.append(-float(cand_score))
                        C_feas_list.append(1.0 if cand_feas else 0.0)

                        improved = (cand_score > score_before + 1e-3)
                        self._adjust_trust_region(improved, is_feasible=cand_feas)

                        if self.eval_count >= self.max_evals:
                            break
                        continue

            # 2b. MALIBO Laplace adaptation + 70/30 early-stopped residual GBDT
            y_clean = np.minimum(np.array(Y_minimize_list, dtype=np.float64), 1.0)
            y_mean = float(np.mean(y_clean))
            y_std_val = max(float(np.std(y_clean)), 1e-4)
            y_norm = (y_clean - y_mean) / y_std_val
            gamma_t = (
                annealed_lfbo_gamma(step, self.n_init, self.max_evals, self.gamma, self.gamma_end)
                if self.use_gamma_annealing
                else self.gamma
            )
            tau_std = float(np.quantile(y_norm, gamma_t))

            x_obs_np = np.array(X_unit_list, dtype=np.float64)
            c_obs_np = np.array(C_feas_list, dtype=np.float64)
            x_obs_tensor = torch.from_numpy(x_obs_np.astype(np.float32)).to(self.device)
            y_obs_tensor = torch.from_numpy(y_norm.astype(np.float32)).to(self.device)

            z_hat, _z_map, _L_prec = self.adapter.adapt_and_sample(
                x_obs=x_obs_tensor,
                y_obs_std=y_obs_tensor,
                tau_std=tau_std,
                rng=self.rng,
                temperature=self.temperature,
            )
            self.residual_booster.fit_with_early_stopping(
                x_obs=x_obs_np,
                y_std=y_norm,
                tau_std=tau_std,
                z_sample=z_hat,
                step_seed=int(self.seed * 1009 + step),
            )

            # 2c. Candidate generation: Trust-Region vs Global + optional 30% LLM directional injection
            x_best_u = self.to_unit(self.best_x)
            n_cand = self.n_candidates

            if self.use_trust_region:
                lb_tr = np.clip(x_best_u - self.L / 2.0, 0.0, 1.0)
                ub_tr = np.clip(x_best_u + self.L / 2.0, 0.0, 1.0)
                cand_raw = self.rng.uniform(lb_tr, ub_tr, size=(n_cand, d))
                mask = self.rng.rand(n_cand, d) < self.prob_perturb
                mask_all_false = ~np.any(mask, axis=1)
                if np.any(mask_all_false):
                    rand_dims = self.rng.randint(0, d, size=int(np.sum(mask_all_false)))
                    mask[mask_all_false, rand_dims] = True
                candidates_u = np.where(mask, cand_raw, x_best_u)
                clip_lb, clip_ub = lb_tr, ub_tr
                dir_scale = self.L
            else:
                candidates_u = self.rng.uniform(0.0, 1.0, size=(n_cand, d))
                clip_lb, clip_ub = 0.0, 1.0
                dir_scale = self.global_dir_ref_scale

            if (
                self.use_llm
                and self.guide_direction_unit is not None
                and self.guide_direction_steps_left > 0
                and step <= 195
            ):
                n_dir = min(1536, n_cand // 3)
                step_scale = self.rng.uniform(0.05 * dir_scale, 0.65 * dir_scale, size=(n_dir, 1))
                dir_cands = (
                    x_best_u
                    + step_scale * self.guide_direction_unit
                    + self.rng.normal(0.0, 0.03 * dir_scale, size=(n_dir, d))
                )
                candidates_u[:n_dir] = np.clip(dir_cands, clip_lb, clip_ub)
                self.guide_direction_steps_left -= 1

            # 2d. Maximize MALIBO acquisition (Constrained/OOD-Aware c-LFBO when enabled)
            raw_utilities = self.residual_booster.predict_utility(
                x_cand=candidates_u,
                z_sample=z_hat,
                use_raw_margin=True,
            )
            if self.use_feas_surrogate:
                acquisition_scores = compute_constrained_lfbo_acquisition(
                    raw_utilities=raw_utilities,
                    X_obs=x_obs_np,
                    c_obs=c_obs_np,
                    X_cand=candidates_u,
                    n_init=self.n_init,
                    lengthscale=self.feas_lengthscale,
                    tau_feas=self.feas_tau,
                    beta=self.feas_beta,
                )
            else:
                acquisition_scores = raw_utilities

            x_next_u = candidates_u[int(np.argmax(acquisition_scores))]
            x_next = self.from_unit(x_next_u)

            new_score = self.evaluate(x_next)
            new_feas = (new_score > 0.0)
            X_unit_list.append(x_next_u)
            Y_minimize_list.append(-float(new_score))
            C_feas_list.append(1.0 if new_feas else 0.0)

            improved = (new_score > score_before + 1e-3)
            self._adjust_trust_region(improved, is_feasible=new_feas)

        elapsed = time.time() - t0
        det = self.evaluator.evaluate_detailed(self.best_x)
        post_init_feas = C_feas_list[self.n_init :]

        out: Dict[str, Any] = {
            "method": self.variant_name,
            "seed": self.seed,
            "best_score": float(self.best_score),
            "history": [float(v) for v in self.history_best[: self.max_evals]],
            "best_x": [float(v) for v in self.best_x],
            "post_init_feasibility_rate": float(np.mean(post_init_feas)) if post_init_feas else 1.0,
            "detailed": {
                "power_kW": float(det.get("power_kW", 0.0)),
                "power_MW": float(det.get("power_MW", 0.0)),
                "torque_Nm": float(det.get("torque_Nm", 0.0)),
                "torque_kNm": float(det.get("torque_kNm", 0.0)),
                "thrust_kN": float(det.get("thrust_kN", 0.0)),
                "Cp": float(det.get("Cp", 0.0)),
                "Ct": float(det.get("Ct", 0.0)),
                "is_feasible": bool(det.get("is_feasible", det.get("feasible", False))),
                "total_penalty": float(det.get("total_penalty", det.get("total_penalty_kW", 0.0))),
                "span_radii": [float(v) for v in det.get("span_radii", [])],
                "span_chord": [float(v) for v in det.get("span_chord", [])],
                "span_twist_deg": [float(v) for v in det.get("span_twist_deg", [])],
                "span_alpha_deg": [float(v) for v in det.get("span_alpha_deg", [])],
                "blade_pitch_deg": float(det.get("blade_pitch_deg", 0.0)),
                "span_effective_pitch_deg": [float(v) for v in det.get("span_effective_pitch_deg", [])],
            },
            "elapsed_seconds": float(elapsed),
        }
        if self.use_llm:
            out["intervention_history"] = self.intervention_history
        return out
