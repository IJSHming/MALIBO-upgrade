# MALIBO-Upgrade: Trust-Region & LLM-Guided Likelihood-Free Bayesian Meta-Learning for High-Dimensional Aerodynamic Shape Optimization

A 100% self-contained, decoupled ablation benchmark evaluating two principled upgrades to **MALIBO** (Pan et al., ICML 2024) across both **2D Constrained Airfoil Design ($D=18$)** and **3D NREL Phase VI Wind Turbine Blade Design ($D=63$)** under a strict 300-evaluation budget (`NFE = 300`, 10 independent seeds, zero data leakage, zero heuristic fallback).

## 1. $2 \times 2$ Factorial Ablation Design

All three upgraded MALIBO variants reuse the exact pretrained zero-leakage MALIBO ResFFN checkpoints (`weights/malibo_2d_checkpoint.pt` trained on 250 2D tasks; `weights/malibo_3d_checkpoint.pt` trained on 100 3D tasks), identical 10 random seeds (`42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999`), and bit-for-bit identical `n_init = 15` initializations against frozen baselines (`baselines/baseline_2d_results.json`, `baselines/baseline_3d_results.json`):

| Method | Surrogate | Adaptive Trust Region (TuRBO-1 style) | Stagnation-Adaptive LLM Offset Injection | Source |
| :--- | :--- | :---: | :---: | :--- |
| **TuRBO-1** (`Vanilla-TuRBO`) | Matérn-5/2 GP | Yes | No | Frozen baseline |
| **MALIBO** (Paper-Exact) | Meta-ResFFN + Laplace TS + GBDT | No (Global $[0,1]^D$) | No | Frozen baseline |
| **TR-MALIBO** | Meta-ResFFN + Laplace TS + GBDT | **Yes** | No | **Upgrade 1 (TR only)** |
| **LLM-MALIBO** | Meta-ResFFN + Laplace TS + GBDT | No (Global $[0,1]^D$) | **Yes** | **Upgrade 2 (LLM only)** |
| **LLM-TR-MALIBO** | Meta-ResFFN + Laplace TS + GBDT | **Yes** | **Yes** | **Upgrade 3 (TR + LLM)** |
| **LLM-Meta-TuRBO** | Matérn-5/2 GP | Yes | Yes | Frozen baseline |

### Key Algorithmic Upgrades
1. **Trust-Region Candidate Localization (`TR-MALIBO`)**: Replaces Paper-Exact MALIBO's global uniform 5,120-point search over $[0,1]^D$ with a TuRBO-1 adaptive local trust region $[x_{\text{best}} \pm L/2]$ and sparse coordinate perturbation probability $p_{\text{perturb}} = \min(20/D, 1)$ ($L_0=0.8$, $L_{\min}=0.5^7$, $L_{\max}=1.6$, $\tau_s=3$, $\tau_f=4$ in 2D / $6$ in 3D), resolving both late-stage exploitation saturation in 18D and the $0/5{,}120$ feasibility collapse in 63D.
2. **Zero-Leakage LLM Directional Offset Injection (`LLM-MALIBO` & `LLM-TR-MALIBO`)**: Couples MALIBO with a local live LLM meta-controller (`Qwen/Qwen3.8-27B-FP8` via SGLang, non-thinking direct JSON mode, zero fallback) triggered at Step 16 (`post_init`) and genuine stagnation ($\ge 10$ steps, interval $\ge 18$, step $\le 180$), proposing smooth orthogonal physical deformation modes (4 symmetric modes in 2D; 12 Root/Mid/Tip modes in 3D), analytical zero-NFE geometry backtracking, 30% directional candidate injection (15 steps), temporary TR modulation (in `LLM-TR-MALIBO`), and Laplace posterior Thompson-sampling temperature $\hat{\mathbf{z}} = \mathbf{z}_{\text{MAP}} + \tau_{\text{TS}}\mathbf{L}^{-T}\bm{\epsilon}$.

---

## 2. Main Experimental Results (10 Paired Seeds, 100% Feasible)

### Case 1: 2D Constrained Airfoil (NACA 0012, $Re = 3.5\times 10^6, \alpha = 3.5^\circ$, $D=18$, $\text{NFE}=300$)

| Method | Step-15 | Final Mean ± SD ($L/D$) | Median [Min, Max] | NFE $\ge 200$ | $\Delta$ vs. MALIBO | $\Delta$ vs. TuRBO-1 | Avg LLM Calls |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla-TuRBO | 72.47 ± 2.24 | $195.73 \pm 12.32$ | 198.97 [164.2, 207.6] | -- | $+15.92$ ($+8.9\%$) | Baseline | -- |
| MALIBO (Orig.) | 72.47 ± 2.24 | $179.81 \pm 9.29$ | 180.00 [166.7, 192.9] | -- | Baseline | $-15.92$ ($-8.1\%$) | -- |
| **LLM-MALIBO** | 72.47 ± 2.24 | $187.36 \pm 12.95$ | 183.38 [166.8, 212.2] | -- | $+7.56$ ($+4.2\%$) | $-8.36$ ($-4.3\%$) | 8.7 |
| **TR-MALIBO** | 72.47 ± 2.24 | **$229.62 \pm 12.70$** | 232.40 [211.2, 245.9] | 100 | **$+49.81$ ($+27.7\%$)** ($p=10^{-5}$) | **$+33.89$ ($+17.3\%$)** ($p=4.3\times 10^{-4}$) | 0 |
| LLM-Meta-TuRBO | 72.47 ± 2.24 | $227.40 \pm 13.41$ | 225.10 [200.8, 247.1] | 120 | $+47.59$ ($+26.5\%$) | $+31.67$ ($+16.2\%$) | 5.1 |
| **LLM-TR-MALIBO** | 72.47 ± 2.24 | **$235.94 \pm 10.95$** | **237.58** [206.5, 248.2] | **99** | **$+56.13$ ($+31.2\%$)** ($p<10^{-5}$) | **$+40.21$ ($+20.5\%$)** ($p=2\times 10^{-5}$) | **2.9** |

### Case 2: 3D NREL Phase VI Wind Turbine Blade ($V_\infty = 7.0\,\text{m/s}$, $72\,\text{rpm}$, $D=63$, Baseline $= 6.0884\,\text{kW}$, $\text{NFE}=300$)

| Method | Step-15 (kW) | Power Mean ± SD (kW) | Median [Min, Max] | Gain vs. Base | NFE $\ge 6.50$ | Avg Thrust (kN) | Avg LLM Calls |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TuRBO-1 | 6.1773 ± 0.0280 | $6.5920 \pm 0.0463$ | 6.5717 [6.507, 6.653] | $+8.27\%$ | 68 | 1.3938 | -- |
| MALIBO (Orig.) | 6.1773 ± 0.0280 | $6.1773 \pm 0.0280$ | 6.1812 [6.125, 6.228] | $+1.46\%$ | -- | 1.2417 | -- |
| **LLM-MALIBO** | 6.1773 ± 0.0280 | $6.6464 \pm 0.0595$ | 6.6742 [6.500, 6.706] | $+9.16\%$ | 52 | 1.3788 | 9.7 |
| **TR-MALIBO** | 6.1773 ± 0.0280 | **$6.6575 \pm 0.0291$** | 6.6653 [6.607, 6.693] | $+9.35\%$ | 60 | 1.3946 | 0 |
| LLM-Meta-TuRBO | 6.1773 ± 0.0280 | $6.6664 \pm 0.0196$ | 6.6573 [6.646, 6.696] | $+9.49\%$ | **36** | 1.3948 | 7.2 |
| **LLM-TR-MALIBO** | 6.1773 ± 0.0280 | **$6.6689 \pm 0.0199$** | **6.6665** [6.642, **6.706**] | **$+9.53\%$** | 39 | 1.3940 | 7.3 |

---

## 3. Repository Layout & Reproduction

```text
MALIBO-upgrade/
├── baselines/            # Frozen 10-seed 2D & 3D baseline JSONs (TuRBO-1, MALIBO, LLM-Meta-TuRBO)
├── weights/              # Pretrained zero-leakage 2D (18D) & 3D (63D) MALIBO ResFFN checkpoints
├── data/                 # NREL Phase VI 31-section full-cosine baseline (.npz)
├── src/case_2d/          # Decoupled 2D evaluator, MALIBO core, LLM agent, and upgraded optimizers
├── src/case_3d/          # Decoupled 3D evaluator, parameterization, MALIBO core, LLM agent, and upgraded optimizers
├── run_benchmark_2d.py   # 10-seed parallel 2D ablation runner
├── run_benchmark_3d.py   # 10-seed parallel 3D ablation runner
├── plot_ablation.py      # Publication-grade figures & paired statistical report generator
├── results/              # Unified 6-method JSON/NPZ results & ablation_summary_report.md
└── figures/              # Vector PDF & 300 DPI PNG figures (fig1–fig5)
```

### Reproduce Commands
```bash
# 1. Start local SGLang server (Qwen/Qwen3.8-27B-FP8 on port 30000)
bash launch_sglang.sh

# 2. Run 2D & 3D 10-seed ablation benchmarks (only runs TR-MALIBO, LLM-MALIBO, LLM-TR-MALIBO)
python run_benchmark_2d.py --n_workers 5 --device cpu
python run_benchmark_3d.py --n_workers 5 --device cpu

# 3. Generate figures and full statistical tests
python plot_ablation.py
```
