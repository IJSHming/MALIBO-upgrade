# MALIBO Upgrade Ablation Study — Complete Statistical Report

## 1. Experimental Matrix (2×2 Factorial Ablation + TuRBO Baselines)

| Method | Surrogate | Trust Region | LLM Offset Injection | Source |
| :--- | :--- | :---: | :---: | :--- |
| **Vanilla-TuRBO / TuRBO-1** | GP (Matérn-5/2) | Yes | No | Frozen baseline |
| **MALIBO (Paper-Exact)** | Meta-ResFFN + Laplace TS + GBDT | No (Global) | No | Frozen baseline |
| **TR-MALIBO** | Meta-ResFFN + Laplace TS + GBDT | **Yes** | No | **New (Upgrade 1)** |
| **LLM-MALIBO** | Meta-ResFFN + Laplace TS + GBDT | No (Global) | **Yes** | **New (Upgrade 2)** |
| **LLM-TR-MALIBO** | Meta-ResFFN + Laplace TS + GBDT | **Yes** | **Yes** | **New (Upgrade 3)** |
| **LLM-Meta-TuRBO** | GP (Matérn-5/2) | Yes | Yes | Frozen baseline |

## 2. 2D Constrained Airfoil Benchmark (NACA 0012, Re=3.5×10⁶, α=3.5°, D=18, NFE=300, 10 Seeds)

| Method | Step-15 | Final Mean ± SD (L/D) | Median [Min, Max] | NFE≥120 | NFE≥160 | NFE≥200 | Feasible | Avg Calls | Time/Seed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla-TuRBO** | 72.47±2.24 | **195.73 ± 12.32** | 198.97 [164.2, 207.6] | 69 | 140 | -- | 10/10 | -- | -- |
| **MALIBO** | 72.47±2.24 | **179.81 ± 9.29** | 180.00 [166.7, 192.9] | 31 | 114 | -- | 10/10 | -- | 66.6s |
| **TR-MALIBO** | 72.47±2.24 | **229.62 ± 12.70** | 232.40 [211.2, 245.9] | 32 | 59 | 100 | 10/10 | -- | 85.7s |
| **LLM-MALIBO** | 72.47±2.24 | **187.36 ± 12.95** | 183.38 [166.8, 212.2] | 29 | 58 | -- | 10/10 | 8.7 | 126.2s |
| **LLM-TR-MALIBO** | 72.47±2.24 | **235.94 ± 10.95** | 237.58 [206.5, 248.2] | 33 | 55 | 99 | 10/10 | 2.9 | 101.3s |
| **LLM-Meta-TuRBO** | 72.47±2.24 | **227.40 ± 13.41** | 225.10 [200.8, 247.1] | 18 | 40 | 120 | 10/10 | 5.1 | 39.4s |

### 2D Paired Statistical Tests (10 Paired Seeds)

| Comparison | Δ Mean (L/D) | Rel. % | Win Rate | Paired t | p (t-test) | p (Wilcoxon) | Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TR-MALIBO vs. MALIBO | +49.81 | +27.70% | 10/10 | +9.295 | 0.00001 | 0.00195 | +2.94 |
| LLM-MALIBO vs. MALIBO | +7.56 | +4.20% | 7/10 | +1.736 | 0.11661 | 0.13086 | +0.55 |
| LLM-TR-MALIBO vs. MALIBO | +56.13 | +31.22% | 10/10 | +9.663 | 0.00000 | 0.00195 | +3.06 |
| TR-MALIBO vs. Vanilla-TuRBO | +33.89 | +17.32% | 10/10 | +5.412 | 0.00043 | 0.00195 | +1.71 |
| LLM-TR-MALIBO vs. TR-MALIBO | +6.32 | +2.75% | 9/10 | +1.018 | 0.33515 | 0.08398 | +0.32 |
| LLM-TR-MALIBO vs. LLM-MALIBO | +48.57 | +25.93% | 10/10 | +8.230 | 0.00002 | 0.00195 | +2.60 |
| LLM-TR-MALIBO vs. Vanilla-TuRBO | +40.21 | +20.54% | 10/10 | +7.947 | 0.00002 | 0.00195 | +2.51 |
| LLM-TR-MALIBO vs. LLM-Meta-TuRBO | +8.53 | +3.75% | 7/10 | +1.620 | 0.13966 | 0.23242 | +0.51 |

## 3. 3D NREL Phase VI Wind Turbine Blade Benchmark (V=7.0 m/s, 72 rpm, D=63, NFE=300, Baseline=6.0884 kW, 10 Seeds)

| Method | Step-15 (kW) | Power Mean ± SD (kW) | Median [Min, Max] | Gain vs. Base | NFE≥6.50 | NFE≥6.60 | Thrust (kN) | Feasible | Avg Calls | Time/Seed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TuRBO** | 6.1773±0.0280 | **6.5920 ± 0.0463** | 6.5717 [6.507, 6.653] | +8.27% | 68 | -- | 1.3938 | 10/10 | -- | 70.9s |
| **MALIBO** | 6.1773±0.0280 | **6.1773 ± 0.0280** | 6.1812 [6.125, 6.228] | +1.46% | -- | -- | 1.2417 | 10/10 | -- | 111.0s |
| **TR-MALIBO** | 6.1773±0.0280 | **6.6575 ± 0.0291** | 6.6653 [6.607, 6.693] | +9.35% | 60 | 115 | 1.3946 | 10/10 | -- | 168.4s |
| **LLM-MALIBO** | 6.1773±0.0280 | **6.6464 ± 0.0595** | 6.6742 [6.500, 6.706] | +9.16% | 52 | 73 | 1.3788 | 10/10 | 9.7 | 245.5s |
| **LLM-TR-MALIBO** | 6.1773±0.0280 | **6.6689 ± 0.0199** | 6.6665 [6.642, 6.706] | +9.53% | 39 | 88 | 1.3940 | 10/10 | 7.3 | 243.6s |
| **LLM-Meta-TuRBO** | 6.1773±0.0280 | **6.6664 ± 0.0196** | 6.6573 [6.646, 6.696] | +9.49% | 36 | 60 | 1.3948 | 10/10 | 7.2 | 157.9s |

### 3D Paired Statistical Tests (10 Paired Seeds)

| Comparison | Δ Power (kW) | Rel. % | Win Rate | Paired t | p (t-test) | p (Wilcoxon) | Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TR-MALIBO vs. MALIBO | +0.4802 | +7.77% | 10/10 | +47.646 | 0.00000 | 0.00195 | +15.07 |
| LLM-MALIBO vs. MALIBO | +0.4691 | +7.59% | 10/10 | +25.064 | 0.00000 | 0.00195 | +7.93 |
| LLM-TR-MALIBO vs. MALIBO | +0.4916 | +7.96% | 10/10 | +40.185 | 0.00000 | 0.00195 | +12.71 |
| TR-MALIBO vs. TuRBO | +0.0655 | +0.99% | 8/10 | +3.836 | 0.00399 | 0.01367 | +1.21 |
| LLM-TR-MALIBO vs. TR-MALIBO | +0.0114 | +0.17% | 6/10 | +1.295 | 0.22752 | 0.27539 | +0.41 |
| LLM-TR-MALIBO vs. LLM-MALIBO | +0.0225 | +0.34% | 6/10 | +1.385 | 0.19942 | 0.37500 | +0.44 |
| LLM-TR-MALIBO vs. TuRBO | +0.0769 | +1.17% | 9/10 | +4.298 | 0.00200 | 0.00391 | +1.36 |
| LLM-TR-MALIBO vs. LLM-Meta-TuRBO | +0.0025 | +0.04% | 5/10 | +0.352 | 0.73325 | 0.84570 | +0.11 |
