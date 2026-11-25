"""Publication-Grade 6-Method Ablation Visualization & Statistical Analysis Script.

Generates high-resolution vector PDF + 300 DPI PNG figures and a comprehensive
markdown statistical report (`results/ablation_summary_report.md`) comparing:
  1. Vanilla-TuRBO / TuRBO-1  (Baseline: GP + TR)
  2. MALIBO (Paper-Exact)     (Baseline: Meta-ResFFN + Laplace TS + GBDT, Global)
  3. TR-MALIBO                (Upgrade 1: + Trust Region)
  4. LLM-MALIBO               (Upgrade 2: + LLM Offset Injection)
  5. LLM-TR-MALIBO            (Upgrade 3: + Trust Region + LLM)
  6. LLM-Meta-TuRBO           (Strong Baseline: GP + TR + LLM)

Covers both 2D Constrained Airfoil (D=18) and 3D NREL Phase VI Wind Turbine Blade (D=63).
"""

import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

STYLE_2D = {
    "Vanilla-TuRBO":  {"label": "TuRBO-1",          "color": "#4C72B0", "ls": "--", "lw": 1.8, "zorder": 3},
    "MALIBO":         {"label": "MALIBO (Orig.)",   "color": "#C44E52", "ls": ":",  "lw": 2.0, "zorder": 3},
    "TR-MALIBO":      {"label": "TR-MALIBO (+TR)",  "color": "#DD8452", "ls": "-",  "lw": 2.2, "zorder": 4},
    "LLM-MALIBO":     {"label": "LLM-MALIBO (+LLM)","color": "#8172B3", "ls": "-.", "lw": 2.1, "zorder": 4},
    "LLM-TR-MALIBO":  {"label": "LLM-TR-MALIBO",    "color": "#1B9E77", "ls": "-",  "lw": 2.6, "zorder": 6},
    "LLM-Meta-TuRBO": {"label": "LLM-Meta-TuRBO",   "color": "#2E6F40", "ls": "--", "lw": 2.1, "zorder": 5},
}

STYLE_3D = {
    "TuRBO":          {"label": "TuRBO-1",          "color": "#4C72B0", "ls": "--", "lw": 1.8, "zorder": 3},
    "MALIBO":         {"label": "MALIBO (Orig.)",   "color": "#C44E52", "ls": ":",  "lw": 2.0, "zorder": 3},
    "TR-MALIBO":      {"label": "TR-MALIBO (+TR)",  "color": "#DD8452", "ls": "-",  "lw": 2.2, "zorder": 4},
    "LLM-MALIBO":     {"label": "LLM-MALIBO (+LLM)","color": "#8172B3", "ls": "-.", "lw": 2.1, "zorder": 4},
    "LLM-TR-MALIBO":  {"label": "LLM-TR-MALIBO",    "color": "#1B9E77", "ls": "-",  "lw": 2.6, "zorder": 6},
    "LLM-Meta-TuRBO": {"label": "LLM-Meta-TuRBO",   "color": "#2E6F40", "ls": "--", "lw": 2.1, "zorder": 5},
}

SHORT_TICK = {
    "Vanilla-TuRBO":  "TuRBO-1",
    "TuRBO":          "TuRBO-1",
    "MALIBO":         "MALIBO\n(Orig.)",
    "TR-MALIBO":      "TR-\nMALIBO",
    "LLM-MALIBO":     "LLM-\nMALIBO",
    "LLM-TR-MALIBO":  "LLM-TR-\nMALIBO",
    "LLM-Meta-TuRBO": "LLM-Meta-\nTuRBO",
}

ORDER_2D = ["Vanilla-TuRBO", "MALIBO", "TR-MALIBO", "LLM-MALIBO", "LLM-TR-MALIBO", "LLM-Meta-TuRBO"]
ORDER_3D = ["TuRBO", "MALIBO", "TR-MALIBO", "LLM-MALIBO", "LLM-TR-MALIBO", "LLM-Meta-TuRBO"]


def paired_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    diff = a - b
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    wins = int(np.sum(diff > 1e-9))
    if std_d < 1e-12:
        t_stat, p_t, d_cohen = 0.0, 1.0, 0.0
    else:
        t_stat, p_t = stats.ttest_rel(a, b)
        d_cohen = mean_d / std_d
    try:
        nonzero = diff[np.abs(diff) > 1e-12]
        if len(nonzero) == 0:
            p_w = 1.0
        else:
            _, p_w = stats.wilcoxon(a, b, zero_method="wilcox")
    except Exception:
        p_w = 1.0
    return {
        "delta": mean_d,
        "wins": wins,
        "n": len(a),
        "t": float(t_stat),
        "p_t": float(p_t),
        "p_w": float(p_w),
        "cohen_d": float(d_cohen),
    }


def first_nfe_reaching(mean_curve: np.ndarray, threshold: float) -> str:
    idx = np.where(mean_curve >= threshold)[0]
    return str(int(idx[0] + 1)) if len(idx) > 0 else "--"


def plot_case_perf(
    results: Dict[str, List[Dict[str, Any]]],
    order: List[str],
    styles: Dict[str, Dict[str, Any]],
    ylabel: str,
    title_left: str,
    title_right: str,
    out_prefix: str,
    baseline_val: float = None,
    use_sem: bool = False,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.3), gridspec_kw={"width_ratios": [1.45, 1.0]})

    x = np.arange(1, 301)
    if baseline_val is not None:
        ax1.axhline(baseline_val, color="#555555", ls=":", lw=1.2, alpha=0.8, label=f"Baseline ({baseline_val:.3f})")

    ax1.axvline(15, color="#888888", ls=":", lw=1.0, alpha=0.7)
    ax1.text(17, 0.04, r"$n_{\mathrm{init}}=15$", transform=ax1.get_xaxis_transform(), fontsize=8, color="#666666")

    box_data = []
    box_labels = []
    box_colors = []

    for m in order:
        rows = results.get(m, [])
        if not rows:
            continue
        st = styles[m]
        hist = np.array([r["history"][:300] for r in rows], dtype=np.float64)
        mean_h = np.mean(hist, axis=0)
        band = np.std(hist, axis=0) / np.sqrt(len(rows)) if use_sem else np.std(hist, axis=0)

        final_mean = float(np.mean(hist[:, -1]))
        final_sd = float(np.std(hist[:, -1]))
        fmt = f"{final_mean:.3f}±{final_sd:.3f}" if final_mean < 20 else f"{final_mean:.1f}±{final_sd:.1f}"
        ax1.plot(x, mean_h, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=st["zorder"],
                 label=f"{st['label']} ({fmt})")
        ax1.fill_between(x, mean_h - band, mean_h + band, color=st["color"], alpha=0.11, zorder=st["zorder"] - 2)

        box_data.append(hist[:, -1])
        box_labels.append(SHORT_TICK[m])
        box_colors.append(st["color"])

    ax1.set_xlim(1, 300)
    ax1.set_xlabel("Function Evaluations (NFE)")
    ax1.set_ylabel(ylabel)
    ax1.set_title(title_left)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    positions = np.arange(1, len(box_data) + 1)
    bp = ax2.boxplot(
        box_data,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "lw": 1.6},
        whiskerprops={"color": "#555555", "lw": 1.1},
        capprops={"color": "#555555", "lw": 1.1},
    )
    rng_j = np.random.RandomState(42)
    for i, (patch, vals, col) in enumerate(zip(bp["boxes"], box_data, box_colors)):
        patch.set_facecolor(col)
        patch.set_alpha(0.32)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.3)

        jitter = rng_j.uniform(-0.14, 0.14, size=len(vals))
        ax2.scatter(positions[i] + jitter, vals, color=col, edgecolor="white", linewidth=0.5, s=26, zorder=4, alpha=0.9)
        mean_v = float(np.mean(vals))
        ax2.scatter([positions[i]], [mean_v], marker="D", color="white", edgecolor=col, linewidth=1.6, s=42, zorder=5)

        val_str = f"{mean_v:.3f}" if mean_v < 20 else f"{mean_v:.1f}"
        y_top = float(np.max(vals))
        span = float(np.max([np.max(b) for b in box_data]) - np.min([np.min(b) for b in box_data]))
        ax2.text(
            positions[i],
            y_top + 0.035 * max(span, 1e-3),
            val_str,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=col,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, lw=0.8, alpha=0.92),
        )

    y_min_all = float(np.min([np.min(b) for b in box_data]))
    y_max_all = float(np.max([np.max(b) for b in box_data]))
    pad = max((y_max_all - y_min_all) * 0.16, 0.05)
    ax2.set_ylim(y_min_all - pad * 0.4, y_max_all + pad * 1.15)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(box_labels, fontsize=7.8)
    ax2.set_ylabel(f"Final {ylabel} (NFE = 300)")
    ax2.set_title(title_right)

    plt.tight_layout()
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_prefix}.pdf and .png")


def plot_factorial_ablation_bar(d2: Dict[str, Any], d3: Dict[str, Any], out_prefix: str):
    """2-panel factorial 2x2 ablation summary (MALIBO -> TR-MALIBO / LLM-MALIBO -> LLM-TR-MALIBO + TuRBO refs)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.0))

    for ax, res, order, styles, ylabel, title, fmt in [
        (ax1, d2["results"], ORDER_2D, STYLE_2D, "Final Best Feasible L/D (Mean ± SD)", "(a) 2D Airfoil Optimization (D = 18, 10 Seeds)", "{:.1f}"),
        (ax2, d3["results"], ORDER_3D, STYLE_3D, "Final Best Rotor Power (kW, Mean ± SD)", "(b) 3D NREL Phase VI Blade (D = 63, 10 Seeds)", "{:.3f}"),
    ]:
        means, sds, labels, colors = [], [], [], []
        for m in order:
            rows = res.get(m, [])
            if not rows:
                continue
            sc = np.array([r["best_score"] for r in rows])
            means.append(float(np.mean(sc)))
            sds.append(float(np.std(sc)))
            labels.append(SHORT_TICK[m])
            colors.append(styles[m]["color"])

        xs = np.arange(len(means))
        bars = ax.bar(xs, means, yerr=sds, capsize=4, width=0.58, color=colors, alpha=0.82, edgecolor="#222222", linewidth=0.8)
        y_lo = min(m - s for m, s in zip(means, sds))
        y_hi = max(m + s for m, s in zip(means, sds))
        span = max(y_hi - y_lo, 0.1)
        ax.set_ylim(y_lo - span * 0.22, y_hi + span * 0.25)

        for b, m_val, s_val in zip(bars, means, sds):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                m_val + s_val + span * 0.03,
                fmt.format(m_val),
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
                color="#222222",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_prefix}.pdf and .png")


def plot_2d_airfoil_profiles(d2: Dict[str, Any], out_prefix: str):
    """Overlay median optimized 2D airfoil shapes across the 6 methods."""
    from src.case_2d.airfoil_utils import get_naca0012_theta, theta_to_coordinates

    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    base_coords = theta_to_coordinates(get_naca0012_theta())
    ax.plot(base_coords["x_closed"], base_coords["y_closed"], color="#444444", ls=":", lw=1.4, label="NACA 0012 (65.8)")

    for m in ORDER_2D:
        rows = d2["results"].get(m, [])
        if not rows:
            continue
        scores = np.array([r["best_score"] for r in rows])
        med_idx = int(np.argsort(scores)[len(scores) // 2])
        row = rows[med_idx]
        coords = row.get("coordinates")
        if coords is None:
            coords = theta_to_coordinates(np.array(row["best_x"]))
        st = STYLE_2D[m]
        ax.plot(
            coords["x_closed"],
            coords["y_closed"],
            color=st["color"],
            ls=st["ls"],
            lw=st["lw"],
            label=f"{st['label']} ({row['best_score']:.1f})",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.09, 0.17)
    ax.set_xlabel("Normalized Chord Location (x/c)")
    ax.set_ylabel("y/c")
    ax.set_title("Representative Median Optimized 2D Airfoil Profiles (Re=3.5e6, AoA=3.5 deg, NFE=300)", pad=36)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True, fontsize=7.5)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_prefix}.pdf and .png")


def plot_3d_blade_spanwise(d3: Dict[str, Any], out_prefix: str):
    """Compare spanwise chord, twist, and effective AoA of median/peak 3D blades across all 6 methods."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    base_det = d3["metadata"].get("baseline_detailed", {})
    r_base = np.array(base_det.get("span_radii", [])) / 5.029
    if len(r_base) > 0:
        axes[0].plot(r_base, base_det["span_chord"], color="#444444", ls="--", lw=1.4, label="Baseline")
        axes[1].plot(r_base, base_det["span_twist_deg"], color="#444444", ls="--", lw=1.4, label="Baseline")
        axes[2].plot(r_base, base_det["span_alpha_deg"], color="#444444", ls="--", lw=1.4, label="Baseline")

    for m in ORDER_3D:
        rows = d3["results"].get(m, [])
        if not rows:
            continue
        best_row = max(rows, key=lambda r: r["best_score"])
        det = best_row["detailed"]
        r_frac = np.array(det["span_radii"]) / 5.029
        st = STYLE_3D[m]
        axes[0].plot(r_frac, det["span_chord"], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
        axes[1].plot(r_frac, det["span_twist_deg"], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
        axes[2].plot(r_frac, det["span_alpha_deg"], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])

    axes[0].set_title("(a) Spanwise Chord c(r/R)")
    axes[0].set_ylabel("Chord c (m)")
    axes[1].set_title("(b) Spanwise Twist β(r/R)")
    axes[1].set_ylabel("Twist β (deg)")
    axes[2].set_title("(c) Effective AoA α(r/R)")
    axes[2].set_ylabel("Effective AoA α (deg)")
    axes[2].axhspan(6.0, 7.5, color="#1B9E77", alpha=0.10)
    for ax in axes:
        ax.set_xlabel("Spanwise Fraction (r/R)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=7, frameon=True, fontsize=8.0)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_prefix}.pdf and .png")


def build_report(d2: Dict[str, Any], d3: Dict[str, Any], out_path: str):
    lines: List[str] = []
    lines.append("# MALIBO Upgrade Ablation Study — Complete Statistical Report\n")
    lines.append("## 1. Experimental Matrix (2×2 Factorial Ablation + TuRBO Baselines)\n")
    lines.append("| Method | Surrogate | Trust Region | LLM Offset Injection | Source |")
    lines.append("| :--- | :--- | :---: | :---: | :--- |")
    lines.append("| **Vanilla-TuRBO / TuRBO-1** | GP (Matérn-5/2) | Yes | No | Frozen baseline |")
    lines.append("| **MALIBO (Paper-Exact)** | Meta-ResFFN + Laplace TS + GBDT | No (Global) | No | Frozen baseline |")
    lines.append("| **TR-MALIBO** | Meta-ResFFN + Laplace TS + GBDT | **Yes** | No | **New (Upgrade 1)** |")
    lines.append("| **LLM-MALIBO** | Meta-ResFFN + Laplace TS + GBDT | No (Global) | **Yes** | **New (Upgrade 2)** |")
    lines.append("| **LLM-TR-MALIBO** | Meta-ResFFN + Laplace TS + GBDT | **Yes** | **Yes** | **New (Upgrade 3)** |")
    lines.append("| **LLM-Meta-TuRBO** | GP (Matérn-5/2) | Yes | Yes | Frozen baseline |\n")

    # 2D Section
    lines.append("## 2. 2D Constrained Airfoil Benchmark (NACA 0012, Re=3.5×10⁶, α=3.5°, D=18, NFE=300, 10 Seeds)\n")
    lines.append("| Method | Step-15 | Final Mean ± SD (L/D) | Median [Min, Max] | NFE≥120 | NFE≥160 | NFE≥200 | Feasible | Avg Calls | Time/Seed |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in ORDER_2D:
        rows = d2["results"][m]
        hist = np.array([r["history"][:300] for r in rows])
        sc = hist[:, -1]
        s15 = hist[:, 14]
        mh = np.mean(hist, axis=0)
        feas = sum(1 for r in rows if r.get("detailed", {}).get("is_feasible", True))
        calls = [len(r.get("interventions", [])) for r in rows if "interventions" in r]
        c_str = f"{np.mean(calls):.1f}" if calls else "--"
        times = [r["elapsed_seconds"] for r in rows if "elapsed_seconds" in r]
        t_str = f"{np.mean(times):.1f}s" if times else "--"
        lines.append(
            f"| **{m}** | {np.mean(s15):.2f}±{np.std(s15):.2f} | **{np.mean(sc):.2f} ± {np.std(sc):.2f}** | "
            f"{np.median(sc):.2f} [{np.min(sc):.1f}, {np.max(sc):.1f}] | "
            f"{first_nfe_reaching(mh, 120)} | {first_nfe_reaching(mh, 160)} | {first_nfe_reaching(mh, 200)} | "
            f"{feas}/{len(rows)} | {c_str} | {t_str} |"
        )

    lines.append("\n### 2D Paired Statistical Tests (10 Paired Seeds)\n")
    lines.append("| Comparison | Δ Mean (L/D) | Rel. % | Win Rate | Paired t | p (t-test) | p (Wilcoxon) | Cohen's d |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    pairs_2d = [
        ("TR-MALIBO", "MALIBO"),
        ("LLM-MALIBO", "MALIBO"),
        ("LLM-TR-MALIBO", "MALIBO"),
        ("TR-MALIBO", "Vanilla-TuRBO"),
        ("LLM-TR-MALIBO", "TR-MALIBO"),
        ("LLM-TR-MALIBO", "LLM-MALIBO"),
        ("LLM-TR-MALIBO", "Vanilla-TuRBO"),
        ("LLM-TR-MALIBO", "LLM-Meta-TuRBO"),
    ]
    for a_name, b_name in pairs_2d:
        a = np.array([r["best_score"] for r in d2["results"][a_name]])
        b = np.array([r["best_score"] for r in d2["results"][b_name]])
        st = paired_stats(a, b)
        rel = st["delta"] / np.mean(b) * 100.0
        lines.append(
            f"| {a_name} vs. {b_name} | {st['delta']:+.2f} | {rel:+.2f}% | {st['wins']}/{st['n']} | "
            f"{st['t']:+.3f} | {st['p_t']:.5f} | {st['p_w']:.5f} | {st['cohen_d']:+.2f} |"
        )

    # 3D Section
    base_p = float(d3["metadata"]["baseline_power_kW"])
    lines.append(f"\n## 3. 3D NREL Phase VI Wind Turbine Blade Benchmark (V=7.0 m/s, 72 rpm, D=63, NFE=300, Baseline={base_p:.4f} kW, 10 Seeds)\n")
    lines.append("| Method | Step-15 (kW) | Power Mean ± SD (kW) | Median [Min, Max] | Gain vs. Base | NFE≥6.50 | NFE≥6.60 | Thrust (kN) | Feasible | Avg Calls | Time/Seed |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in ORDER_3D:
        rows = d3["results"][m]
        hist = np.array([r["history"][:300] for r in rows])
        sc = hist[:, -1]
        s15 = hist[:, 14]
        mh = np.mean(hist, axis=0)
        th = np.array([r["detailed"]["thrust_kN"] for r in rows])
        feas = sum(1 for r in rows if r["detailed"].get("is_feasible", False))
        calls = [len(r.get("intervention_history", [])) for r in rows if "intervention_history" in r]
        c_str = f"{np.mean(calls):.1f}" if calls else "--"
        tm = np.array([r["elapsed_seconds"] for r in rows])
        gain = (np.mean(sc) - base_p) / base_p * 100.0
        lines.append(
            f"| **{m}** | {np.mean(s15):.4f}±{np.std(s15):.4f} | **{np.mean(sc):.4f} ± {np.std(sc):.4f}** | "
            f"{np.median(sc):.4f} [{np.min(sc):.3f}, {np.max(sc):.3f}] | {gain:+.2f}% | "
            f"{first_nfe_reaching(mh, 6.50)} | {first_nfe_reaching(mh, 6.60)} | "
            f"{np.mean(th):.4f} | {feas}/{len(rows)} | {c_str} | {np.mean(tm):.1f}s |"
        )

    lines.append("\n### 3D Paired Statistical Tests (10 Paired Seeds)\n")
    lines.append("| Comparison | Δ Power (kW) | Rel. % | Win Rate | Paired t | p (t-test) | p (Wilcoxon) | Cohen's d |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    pairs_3d = [
        ("TR-MALIBO", "MALIBO"),
        ("LLM-MALIBO", "MALIBO"),
        ("LLM-TR-MALIBO", "MALIBO"),
        ("TR-MALIBO", "TuRBO"),
        ("LLM-TR-MALIBO", "TR-MALIBO"),
        ("LLM-TR-MALIBO", "LLM-MALIBO"),
        ("LLM-TR-MALIBO", "TuRBO"),
        ("LLM-TR-MALIBO", "LLM-Meta-TuRBO"),
    ]
    for a_name, b_name in pairs_3d:
        a = np.array([r["best_score"] for r in d3["results"][a_name]])
        b = np.array([r["best_score"] for r in d3["results"][b_name]])
        st = paired_stats(a, b)
        rel = st["delta"] / np.mean(b) * 100.0
        lines.append(
            f"| {a_name} vs. {b_name} | {st['delta']:+.4f} | {rel:+.2f}% | {st['wins']}/{st['n']} | "
            f"{st['t']:+.3f} | {st['p_t']:.5f} | {st['p_w']:.5f} | {st['cohen_d']:+.2f} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved statistical summary report to {out_path}")


def main():
    os.makedirs("figures", exist_ok=True)
    with open("results/ablation_2d_results.json", "r", encoding="utf-8") as f:
        d2 = json.load(f)
    with open("results/ablation_3d_results.json", "r", encoding="utf-8") as f:
        d3 = json.load(f)

    plot_case_perf(
        results=d2["results"],
        order=ORDER_2D,
        styles=STYLE_2D,
        ylabel="Best Feasible L/D",
        title_left="(a) 2D Airfoil Convergence Trajectories (Mean ± SD, 10 Seeds)",
        title_right="(b) 2D Final Performance Distribution",
        out_prefix="figures/fig1_2d_ablation_perf",
        use_sem=False,
    )
    plot_case_perf(
        results=d3["results"],
        order=ORDER_3D,
        styles=STYLE_3D,
        ylabel="Best Feasible Rotor Power (kW)",
        title_left="(a) 3D NREL Phase VI Convergence (Mean ± SEM, 10 Seeds)",
        title_right="(b) 3D Final Power Distribution",
        out_prefix="figures/fig2_3d_ablation_perf",
        baseline_val=float(d3["metadata"]["baseline_power_kW"]),
        use_sem=True,
    )
    plot_factorial_ablation_bar(d2, d3, "figures/fig3_cross_dim_factorial_ablation")
    plot_2d_airfoil_profiles(d2, "figures/fig4_2d_airfoil_shapes")
    plot_3d_blade_spanwise(d3, "figures/fig5_3d_blade_spanwise")
    build_report(d2, d3, "results/ablation_summary_report.md")


if __name__ == "__main__":
    main()
