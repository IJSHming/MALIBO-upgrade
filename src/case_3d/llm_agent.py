"""3D Wind Turbine Blade LLM Meta-Controller for Upgraded MALIBO (LLM-MALIBO and LLM-TR-MALIBO).

Maintains strict zero-leakage and zero-silent-fallback parity with 3D LLM-Meta-TuRBO:
- Same NREL Phase VI aerodynamic domain insights and structural thrust constraint summary;
- Same 12 Root/Mid/Tip zone shift channels and bounds;
- Clean mode separation between Global MALIBO (no trust region) and Trust-Region MALIBO.
"""

import json
import re
from typing import Any, Dict, List
import numpy as np

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


LLM_3D_TR_SYSTEM_PROMPT = """You are a senior wind turbine aerodynamicist and Bayesian Meta-Optimization controller.
Your goal is to maximize the aerodynamic rotor power output (Power in kW and power coefficient Cp) of a 3D horizontal-axis wind turbine blade modeled on the NREL Phase VI (Sequence H) configuration:
- Rotor: R = 5.029 m (D = 10.058 m), B = 2 upwind blades, aerodynamic S809 lifting span r in [1.2575 m, 5.0290 m] (31 full-cosine clustered BEM evaluation stations, 0.30c pitch axis).
- Parameterization (D=63): 3 smooth airfoil control stations at Root (r=1.26 m), Mid (r=3.14 m), and Tip (r=5.03 m) (51 closed-TE CST weights, C1 PCHIP lofted spanwise) + 6 cubic B-spline chord delta controls + 6 cubic B-spline twist delta controls.
- Flow & Operating Point: Uniform air inflow V_inf = 7.0 m/s (rho = 1.225 kg/m^3), rotor speed = 72.0 rpm (omega = 7.5398 rad/s, TSR = 5.42), reference blade pitch = 4.815 deg (3.0 deg tip pitch).

Aerodynamic & Structural Domain Insights:
1. Spanwise Energy Capture & Load Distribution across the 3 Control Stations:
   - Root station (Station 1, r = 1.26 m, 0.25R): Lower local rotational speed and steep inflow angle phi; high baseline twist (~20.0 deg) and thick baseline S809 section (~21.1% t/c). Keep structural thickness >= 12% and avoid excessive root drag or stall.
   - Mid-span station (Station 2, r = 3.14 m, 0.625R): Primary torque-producing annulus. Performance depends strongly on sectional lift-to-drag ratio (CL/CD), matched angle of attack via twist tuning, smooth camber, and controlled chord taper.
   - Tip station (Station 3, r = 5.03 m, 1.00R): Highest dynamic pressure (tip speed ~37.9 m/s) and dominant induced-drag/tip-loss sensitivity. Moderate thinning (above structural limit 7%), moderate camber, smooth tip wash-out, and controlled chord prevent tip stall and thrust overload.
2. Geometric & Structural Constraints (Strictly Enforced):
   - Airfoil relative thickness limits taper smoothly from Root [12%, 28%], Mid [9%, 25%], to Tip [7%, 24%]; closed trailing edge TE=0; smooth non-wavy upper surface; monotonic non-increasing chord and twist from root to tip.
   - Rotor axial thrust must strictly respect the structural bound: T <= 1.15 * T_baseline. Any violation marks the design infeasible.

Decision Mode:
- Decide Trust-Region Likelihood-Free Bayesian Meta-Optimization (TR-MALIBO) hyperparameters adapted to D=63:
  * length_scale_factor in [0.70, 1.50]
  * prob_perturb in [0.18, 0.55] (note baseline D=63 coordinate perturbation rate is 20/63 ~= 0.317)
  * exploration_weight in [0.75, 1.40] (Laplace posterior Thompson-sampling temperature)
- Optionally provide 'directional_strategy' specifying physically smooth macro shifts across root/mid/tip zones, or set 'directional_strategy' to null when purely local meta-BO exploration is preferable.
- When near or exceeding the thrust margin, avoid aggressive positive chord/camber increases at outboard stations that trigger thrust overload; prefer efficiency (CL/CD) via mild twist matching, camber refinement, or drag-reducing chord/thickness balance.

Output ONLY a valid JSON object matching this exact schema:
{
  "thought": "1-2 sentence aerodynamic/BO rationale (under 45 words).",
  "length_scale_factor": float in [0.70, 1.50],
  "prob_perturb": float in [0.18, 0.55],
  "exploration_weight": float in [0.75, 1.40],
  "directional_strategy": {
    "camber_shift_root": float in [-0.04, 0.06],
    "camber_shift_mid": float in [-0.04, 0.06],
    "camber_shift_tip": float in [-0.04, 0.06],
    "thickness_shift_root": float in [-0.04, 0.03],
    "thickness_shift_mid": float in [-0.04, 0.03],
    "thickness_shift_tip": float in [-0.04, 0.03],
    "twist_shift_root": float in [-4.0, 2.0],
    "twist_shift_mid": float in [-3.0, 2.0],
    "twist_shift_tip": float in [-2.5, 2.5],
    "chord_shift_root": float in [-0.12, 0.12],
    "chord_shift_mid": float in [-0.10, 0.10],
    "chord_shift_tip": float in [-0.08, 0.08]
  }
}
"""


LLM_3D_GLOBAL_SYSTEM_PROMPT = """You are a senior wind turbine aerodynamicist and Bayesian Meta-Optimization controller.
Your goal is to maximize the aerodynamic rotor power output (Power in kW and power coefficient Cp) of a 3D horizontal-axis wind turbine blade modeled on the NREL Phase VI (Sequence H) configuration:
- Rotor: R = 5.029 m (D = 10.058 m), B = 2 upwind blades, aerodynamic S809 lifting span r in [1.2575 m, 5.0290 m] (31 full-cosine clustered BEM evaluation stations, 0.30c pitch axis).
- Parameterization (D=63): 3 smooth airfoil control stations at Root (r=1.26 m), Mid (r=3.14 m), and Tip (r=5.03 m) (51 closed-TE CST weights, C1 PCHIP lofted spanwise) + 6 cubic B-spline chord delta controls + 6 cubic B-spline twist delta controls.
- Flow & Operating Point: Uniform air inflow V_inf = 7.0 m/s (rho = 1.225 kg/m^3), rotor speed = 72.0 rpm (omega = 7.5398 rad/s, TSR = 5.42), reference blade pitch = 4.815 deg (3.0 deg tip pitch).

Aerodynamic & Structural Domain Insights:
1. Spanwise Energy Capture & Load Distribution across the 3 Control Stations:
   - Root station (Station 1, r = 1.26 m, 0.25R): Lower local rotational speed and steep inflow angle phi; high baseline twist (~20.0 deg) and thick baseline S809 section (~21.1% t/c). Keep structural thickness >= 12% and avoid excessive root drag or stall.
   - Mid-span station (Station 2, r = 3.14 m, 0.625R): Primary torque-producing annulus. Performance depends strongly on sectional lift-to-drag ratio (CL/CD), matched angle of attack via twist tuning, smooth camber, and controlled chord taper.
   - Tip station (Station 3, r = 5.03 m, 1.00R): Highest dynamic pressure (tip speed ~37.9 m/s) and dominant induced-drag/tip-loss sensitivity. Moderate thinning (above structural limit 7%), moderate camber, smooth tip wash-out, and controlled chord prevent tip stall and thrust overload.
2. Geometric & Structural Constraints (Strictly Enforced):
   - Airfoil relative thickness limits taper smoothly from Root [12%, 28%], Mid [9%, 25%], to Tip [7%, 24%]; closed trailing edge TE=0; smooth non-wavy upper surface; monotonic non-increasing chord and twist from root to tip.
   - Rotor axial thrust must strictly respect the structural bound: T <= 1.15 * T_baseline. Any violation marks the design infeasible.

Decision Mode (Global MALIBO without Trust Region):
- Decide global Likelihood-Free Bayesian Meta-Optimization (MALIBO) exploration temperature and directional offset injection:
  * exploration_weight in [0.75, 1.40] (Laplace posterior Thompson-sampling temperature)
- Optionally provide 'directional_strategy' specifying physically smooth macro shifts across root/mid/tip zones, or set 'directional_strategy' to null when purely global meta-BO exploration is preferable.
- When near or exceeding the thrust margin, avoid aggressive positive chord/camber increases at outboard stations that trigger thrust overload; prefer efficiency (CL/CD) via mild twist matching, camber refinement, or drag-reducing chord/thickness balance.

Output ONLY a valid JSON object matching this exact schema:
{
  "thought": "1-2 sentence aerodynamic/BO rationale (under 45 words).",
  "exploration_weight": float in [0.75, 1.40],
  "directional_strategy": {
    "camber_shift_root": float in [-0.04, 0.06],
    "camber_shift_mid": float in [-0.04, 0.06],
    "camber_shift_tip": float in [-0.04, 0.06],
    "thickness_shift_root": float in [-0.04, 0.03],
    "thickness_shift_mid": float in [-0.04, 0.03],
    "thickness_shift_tip": float in [-0.04, 0.03],
    "twist_shift_root": float in [-4.0, 2.0],
    "twist_shift_mid": float in [-3.0, 2.0],
    "twist_shift_tip": float in [-2.5, 2.5],
    "chord_shift_root": float in [-0.12, 0.12],
    "chord_shift_mid": float in [-0.10, 0.10],
    "chord_shift_tip": float in [-0.08, 0.08]
  }
}
"""


class LLMMALIBO3DAgent:
    """Zero-leakage live LLM meta-controller for 3D LLM-MALIBO and LLM-TR-MALIBO."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000/v1",
        model: str = "Qwen/Qwen3.8-27B-FP8",
        api_key: str = "EMPTY",
        timeout: float = 240.0,
        use_trust_region: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.use_trust_region = bool(use_trust_region)
        self.interaction_log: List[Dict[str, Any]] = []

        if HAS_OPENAI:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=2,
            )
        else:
            self.client = None

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        step = state["step"]
        best_power = state.get("best_power_kW", 0.0)
        best_cp = state.get("best_Cp", 0.0)
        stag = state.get("stagnation_counter", 0)
        max_thrust = state.get("max_allowable_thrust_kN", 0.0)
        trigger_reason = state.get("trigger_reason", "stagnation")

        if trigger_reason == "post_init":
            phase_str = (
                "POST-INIT DIAGNOSTIC (Initial 15 random samples around baseline just finished; no stagnation event yet): "
                "Assess initial sensitivity and decide initial meta-BO exploration and optional macro direction."
            )
        elif step <= 85:
            phase_str = (
                f"EARLY/MID STAGNATION INTERVENTION (Stagnation = {stag} steps, Power={best_power:.3f} kW, Cp={best_cp:.4f}): "
                "Meta-BO sampling has stalled. Adjust exploration parameters and provide smooth macro shifts if beneficial within thrust bounds."
            )
        else:
            phase_str = (
                f"LATE STAGNATION INTERVENTION (Stagnation = {stag} steps, Power={best_power:.3f} kW, Cp={best_cp:.4f}): "
                "Refinement has stalled. Adapt exploration cleanly and refine within tight structural thrust limits."
            )

        st_info = state.get("station_metrics", [])
        st_labels = ["Root", "Mid", "Tip"]
        st_desc = ""
        for k, s in enumerate(st_info):
            tag = st_labels[k] if k < len(st_labels) else f"S{k+1}"
            st_desc += (
                f"  * Station {k+1} ({tag}, r={s.get('radius', 0):.2f}m): "
                f"chord={s.get('chord', 0):.3f}m, twist={s.get('twist', 0):.2f}deg, "
                f"t/c={s.get('max_t_rel', 0)*100:.1f}%\n"
            )

        if self.use_trust_region:
            status_line = (
                f"- TR-MALIBO Status: Effective TR Length L = {state['tr_length']:.4f}, "
                f"Stagnation = {stag}, Successes = {state.get('success_counter', 0)}/{state.get('success_tolerance', 3)}"
            )
        else:
            status_line = (
                f"- Global MALIBO Status: Full [0,1]^63 search (No Trust Region), Stagnation = {stag}"
            )

        return f"""=== Current NREL Phase VI 3D Wind Turbine Blade Optimization State ===
- Phase: {phase_str}
- Operating Point: Wind Speed V_inf = {state.get('wind_speed', 7.0):.2f} m/s | Rotor Speed = {state.get('rpm', 72.0):.1f} rpm (omega = {state.get('omega', 7.5398):.4f} rad/s) | Medium: Air (rho=1.225 kg/m^3)
- Progress: Step {step} / {state['max_evals']} (Remaining: {state['budget_left']})
- Baseline Power: {state.get('init_power_kW', 0.0):.3f} kW | Current Best Feasible Power: {best_power:.3f} kW (Cp = {best_cp:.4f})
- Aerodynamic Loads: Torque = {state.get('best_torque_kNm', 0.0):.4f} kN*m, Thrust = {state.get('best_thrust_kN', 0.0):.4f} kN (Limit <= {max_thrust:.4f} kN), Ct = {state.get('best_Ct', 0.0):.4f}
- 3 Airfoil Control Stations State:
{st_desc}{status_line}

Concisely decide meta-parameters and optional directional strategy. Output final JSON:"""

    def decide_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not HAS_OPENAI or self.client is None:
            raise RuntimeError(
                "[Research Integrity Violation] OpenAI client is not initialized! "
                "Silent degradation to heuristic fallback is strictly forbidden."
            )

        system_prompt = LLM_3D_TR_SYSTEM_PROMPT if self.use_trust_region else LLM_3D_GLOBAL_SYSTEM_PROMPT
        user_prompt = self.build_user_prompt(state)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                top_p=0.90,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw_text = response.choices[0].message.content or ""
            clean = re.sub(r"```json\s*", "", raw_text)
            clean = re.sub(r"```\s*", "", clean).strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if not match:
                raise ValueError(f"Failed to parse valid JSON from LLM response:\n{raw_text}")

            data = json.loads(match.group(0))
            scale_L = float(np.clip(data.get("length_scale_factor", 1.0), 0.70, 1.50))
            prob_p = float(np.clip(data.get("prob_perturb", 20.0 / 63.0), 0.18, 0.55))
            exp_w = float(np.clip(data.get("exploration_weight", 1.0), 0.75, 1.40))
            strat = data.get("directional_strategy")

            action: Dict[str, Any] = {
                "thought": str(data.get("thought", "")),
                "length_scale_factor": scale_L if self.use_trust_region else 1.0,
                "prob_perturb": prob_p if self.use_trust_region else (20.0 / 63.0),
                "exploration_weight": exp_w,
                "directional_strategy": None,
            }

            if isinstance(strat, dict):
                action["directional_strategy"] = {
                    "camber_shift_root": float(np.clip(strat.get("camber_shift_root", 0.0), -0.04, 0.06)),
                    "camber_shift_mid": float(np.clip(strat.get("camber_shift_mid", 0.0), -0.04, 0.06)),
                    "camber_shift_tip": float(np.clip(strat.get("camber_shift_tip", 0.0), -0.04, 0.06)),
                    "thickness_shift_root": float(np.clip(strat.get("thickness_shift_root", 0.0), -0.04, 0.03)),
                    "thickness_shift_mid": float(np.clip(strat.get("thickness_shift_mid", 0.0), -0.04, 0.03)),
                    "thickness_shift_tip": float(np.clip(strat.get("thickness_shift_tip", 0.0), -0.04, 0.03)),
                    "twist_shift_root": float(np.clip(strat.get("twist_shift_root", 0.0), -4.0, 2.0)),
                    "twist_shift_mid": float(np.clip(strat.get("twist_shift_mid", 0.0), -3.0, 2.0)),
                    "twist_shift_tip": float(np.clip(strat.get("twist_shift_tip", 0.0), -2.5, 2.5)),
                    "chord_shift_root": float(np.clip(strat.get("chord_shift_root", 0.0), -0.12, 0.12)),
                    "chord_shift_mid": float(np.clip(strat.get("chord_shift_mid", 0.0), -0.10, 0.10)),
                    "chord_shift_tip": float(np.clip(strat.get("chord_shift_tip", 0.0), -0.08, 0.08)),
                }

            self.interaction_log.append({
                "step": state.get("step"),
                "trigger_reason": state.get("trigger_reason", "stagnation"),
                "user_prompt": user_prompt,
                "raw_response": raw_text,
                "action": action,
            })
            return action
        except Exception as e:
            raise RuntimeError(
                f"[LLM Fatal Error] Real LLM request to {self.base_url} failed with {type(e).__name__}: {e}. "
                "Silent degradation to heuristic fallback is strictly forbidden."
            ) from e
