"""出版级可视化与统计报告生成脚本 (CoLT-MALIBO 主对比 + 消融实验双分组架构)。

严格遵循以下作图与分组规范：
1. 纯颜色区分 + 全实线（`ls="-"`），不使用虚线/点划线等线型材质，画面清晰整洁；
2. 两组严格分离对比：
   - Group 1（主方法对比）：`TuRBO-1`、`MALIBO (Orig.)`、`LLM-Meta-TuRBO`、`CoLT-MALIBO`（完整升级版）；
   - Group 2（组件消融实验）：`CoLT-MALIBO`（完整版）、`w/o LLM`、`w/o Obj-TR`、`w/o Con-TR`；
3. 2D 算例同时输出【独立收敛曲线】、【独立箱线图】与【合并双栏图】；
   3D 算例因箱线图视觉区分度有限，不绘制箱线图，仅输出清晰的高分辨率收敛曲线图；
4. 新增 3D 算例三控制站位（Root / Mid-Span / Tip）翼型截面剖面对比图（独立版 + 与展向弦长/扭角/有效攻角合并的 2×3 综合图）；
5. 自动生成详尽的中文统计显著性汇总报告 `results/ablation_summary_report.md`。
"""

import json
import os
from typing import Any, Dict, List, Optional

import aerosandbox as asb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.case_2d.airfoil_utils import get_naca0012_theta, theta_to_coordinates
from src.case_3d.blade_geometry import NUM_STATIONS, ROTOR_RADIUS, STATION_RADII
from src.case_3d.parameterizations import BSplineBladeModel

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9.8,
    "axes.titlesize": 10.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.8,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ------------------------------------------------------------------------------
# 纯色实线样式配置（高区分度，无需考虑色盲友好度，零线型纹理杂乱感）
# ------------------------------------------------------------------------------
STYLE_MAIN_2D = {
    "Vanilla-TuRBO":  {"label": "TuRBO-1",        "color": "#2563EB", "lw": 2.1, "zorder": 3},
    "MALIBO":         {"label": "MALIBO (Orig.)", "color": "#DC2626", "lw": 2.1, "zorder": 3},
    "LLM-Meta-TuRBO": {"label": "LLM-Meta-TuRBO", "color": "#9333EA", "lw": 2.2, "zorder": 4},
    "CoLT-MALIBO":    {"label": "CoLT-MALIBO",    "color": "#00A651", "lw": 2.7, "zorder": 6},
}

STYLE_MAIN_3D = {
    "TuRBO":          {"label": "TuRBO-1",        "color": "#2563EB", "lw": 2.1, "zorder": 3},
    "MALIBO":         {"label": "MALIBO (Orig.)", "color": "#DC2626", "lw": 2.1, "zorder": 3},
    "LLM-Meta-TuRBO": {"label": "LLM-Meta-TuRBO", "color": "#9333EA", "lw": 2.2, "zorder": 4},
    "CoLT-MALIBO":    {"label": "CoLT-MALIBO",    "color": "#00A651", "lw": 2.7, "zorder": 6},
}

STYLE_ABLATION = {
    "CoLT-MALIBO": {"label": "CoLT-MALIBO (Full)", "color": "#00A651", "lw": 2.7, "zorder": 6},
    "w/o LLM":     {"label": "w/o LLM",            "color": "#F97316", "lw": 2.2, "zorder": 4},
    "w/o Obj-TR":  {"label": "w/o Obj-TR",         "color": "#0EA5E9", "lw": 2.2, "zorder": 4},
    "w/o Con-TR":  {"label": "w/o Con-TR",         "color": "#E11D48", "lw": 2.2, "zorder": 5},
}

SHORT_TICK = {
    "Vanilla-TuRBO":  "TuRBO-1",
    "TuRBO":          "TuRBO-1",
    "MALIBO":         "MALIBO\n(Orig.)",
    "LLM-Meta-TuRBO": "LLM-Meta-\nTuRBO",
    "CoLT-MALIBO":    "CoLT-\nMALIBO",
    "w/o LLM":        "w/o\nLLM",
    "w/o Obj-TR":     "w/o\nObj-TR",
    "w/o Con-TR":     "w/o\nCon-TR",
}

GROUP1_2D = ["Vanilla-TuRBO", "MALIBO", "LLM-Meta-TuRBO", "CoLT-MALIBO"]
GROUP1_3D = ["TuRBO", "MALIBO", "LLM-Meta-TuRBO", "CoLT-MALIBO"]
GROUP2_ABL = ["CoLT-MALIBO", "w/o LLM", "w/o Obj-TR", "w/o Con-TR"]


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
        p_w = 1.0 if len(nonzero) == 0 else float(stats.wilcoxon(a, b, zero_method="wilcox")[1])
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


def _draw_convergence_panel(
    ax: plt.Axes,
    results: Dict[str, List[Dict[str, Any]]],
    order: List[str],
    styles: Dict[str, Dict[str, Any]],
    ylabel: str,
    title: str,
    baseline_val: Optional[float] = None,
    use_sem: bool = False,
):
    x = np.arange(1, 301)
    if baseline_val is not None:
        ax.axhline(
            baseline_val,
            color="#9CA3AF",
            ls="-",
            lw=1.4,
            alpha=0.85,
            zorder=1,
            label=f"Baseline ({baseline_val:.3f} kW)",
        )

    ax.axvline(15, color="#9CA3AF", ls=":", lw=1.0, alpha=0.75)
    y_lbl = 0.09 if baseline_val is not None else 0.04
    ax.text(18, y_lbl, r"$n_{\mathrm{init}}=15$", transform=ax.get_xaxis_transform(), fontsize=8.2, color="#666666")

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
        ax.plot(
            x,
            mean_h,
            color=st["color"],
            ls="-",
            lw=st["lw"],
            zorder=st["zorder"],
            label=f"{st['label']} ({fmt})",
        )
        ax.fill_between(
            x,
            mean_h - band,
            mean_h + band,
            color=st["color"],
            alpha=0.13,
            zorder=st["zorder"] - 2,
        )

    ax.set_xlim(1, 300)
    ax.set_xlabel("Function Evaluations (NFE)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True, framealpha=0.93, edgecolor="#cccccc")


def _draw_boxplot_panel(
    ax: plt.Axes,
    results: Dict[str, List[Dict[str, Any]]],
    order: List[str],
    styles: Dict[str, Dict[str, Any]],
    ylabel: str,
    title: str,
):
    box_data, box_labels, box_colors = [], [], []
    for m in order:
        rows = results.get(m, [])
        if not rows:
            continue
        hist = np.array([r["history"][:300] for r in rows], dtype=np.float64)
        box_data.append(hist[:, -1])
        box_labels.append(SHORT_TICK[m])
        box_colors.append(styles[m]["color"])

    positions = np.arange(1, len(box_data) + 1)
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F2937", "lw": 1.8},
        whiskerprops={"color": "#4B5563", "lw": 1.2},
        capprops={"color": "#4B5563", "lw": 1.2},
    )

    y_min_all = float(np.min([np.min(b) for b in box_data]))
    y_max_all = float(np.max([np.max(b) for b in box_data]))
    span = max(y_max_all - y_min_all, 1.0)

    rng_j = np.random.RandomState(42)
    for i, (patch, vals, col) in enumerate(zip(bp["boxes"], box_data, box_colors)):
        patch.set_facecolor(col)
        patch.set_alpha(0.30)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.5)

        jitter = rng_j.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(positions[i] + jitter, vals, color=col, edgecolor="white", linewidth=0.6, s=30, zorder=4, alpha=0.92)
        mean_v = float(np.mean(vals))
        ax.scatter([positions[i]], [mean_v], marker="D", color="white", edgecolor=col, linewidth=1.7, s=46, zorder=5)

        val_str = f"{mean_v:.3f}" if mean_v < 20 else f"{mean_v:.1f}"
        ax.text(
            positions[i],
            float(np.max(vals)) + 0.035 * span,
            val_str,
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
            color=col,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=col, lw=0.9, alpha=0.94),
        )

    pad = max(span * 0.16, 0.5)
    ax.set_ylim(y_min_all - pad * 0.45, y_max_all + pad * 1.15)
    ax.set_xticks(positions)
    ax.set_xticklabels(box_labels, fontsize=8.6)
    ax.set_ylabel(f"Final {ylabel} (NFE = 300)")
    ax.set_title(title)


def save_fig(fig: plt.Figure, out_prefix: str):
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_prefix}.pdf and .png")


def plot_2d_group(
    results: Dict[str, List[Dict[str, Any]]],
    order: List[str],
    styles: Dict[str, Dict[str, Any]],
    tag_name: str,
    prefix_conv: str,
    prefix_box: str,
    prefix_comb: str,
):
    """2D 算例同时输出独立收敛图、独立箱线图和双栏合并图。"""
    ylabel = "Best Feasible L/D"

    # 1. Standalone convergence
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    _draw_convergence_panel(
        ax,
        results,
        order,
        styles,
        ylabel=ylabel,
        title=f"2D Airfoil {tag_name}: Convergence (Mean ± SD, 10 Seeds)",
        use_sem=False,
    )
    plt.tight_layout()
    save_fig(fig, prefix_conv)

    # 2. Standalone boxplot
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    _draw_boxplot_panel(
        ax,
        results,
        order,
        styles,
        ylabel=ylabel,
        title=f"2D Airfoil {tag_name}: Final L/D Distribution (10 Seeds)",
    )
    plt.tight_layout()
    save_fig(fig, prefix_box)

    # 3. Combined 2-panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3), gridspec_kw={"width_ratios": [1.38, 1.0]})
    _draw_convergence_panel(
        ax1,
        results,
        order,
        styles,
        ylabel=ylabel,
        title=f"(a) 2D {tag_name} Convergence (Mean ± SD, 10 Seeds)",
        use_sem=False,
    )
    _draw_boxplot_panel(
        ax2,
        results,
        order,
        styles,
        ylabel=ylabel,
        title=f"(b) 2D {tag_name} Final Distribution",
    )
    plt.tight_layout()
    save_fig(fig, prefix_comb)


def plot_3d_convergence_only(
    results: Dict[str, List[Dict[str, Any]]],
    order: List[str],
    styles: Dict[str, Dict[str, Any]],
    baseline_power: float,
    title: str,
    out_prefix: str,
):
    """3D 算例仅输出清晰收敛曲线图（不画箱线图）。"""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _draw_convergence_panel(
        ax,
        results,
        order,
        styles,
        ylabel="Best Feasible Rotor Power (kW)",
        title=title,
        baseline_val=baseline_power,
        use_sem=True,
    )
    plt.tight_layout()
    save_fig(fig, out_prefix)


def plot_2d_airfoil_profiles(d2: Dict[str, Any], out_prefix: str):
    """Group 1 主对比方法在 2D 算例下的中位数代表性最优翼型轮廓（纯色实线）。"""
    fig, ax = plt.subplots(figsize=(7.8, 3.1))
    base_coords = theta_to_coordinates(get_naca0012_theta())
    ax.plot(
        base_coords["x_closed"],
        base_coords["y_closed"],
        color="#9CA3AF",
        ls="-",
        lw=1.6,
        zorder=2,
        label="NACA 0012 (65.8)",
    )

    for m in GROUP1_2D:
        rows = d2["results"].get(m, [])
        if not rows:
            continue
        scores = np.array([r["best_score"] for r in rows])
        med_idx = int(np.argsort(scores)[len(scores) // 2])
        row = rows[med_idx]
        coords = row.get("coordinates") or theta_to_coordinates(np.array(row["best_x"]))
        st = STYLE_MAIN_2D[m]
        ax.plot(
            coords["x_closed"],
            coords["y_closed"],
            color=st["color"],
            ls="-",
            lw=st["lw"],
            zorder=st["zorder"],
            label=f"{st['label']} ({row['best_score']:.1f})",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.09, 0.17)
    ax.set_xlabel("Normalized Chord Location ($x/c$)")
    ax.set_ylabel("$y/c$")
    ax.set_title("Representative Median Optimized 2D Airfoil Profiles (Re = 3.5e6, AoA = 3.5 deg, NFE = 300)", pad=32)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=True, fontsize=7.6)
    plt.tight_layout()
    save_fig(fig, out_prefix)


def _draw_3d_spanwise_axes(axes: np.ndarray, d3: Dict[str, Any]):
    base_det = d3["metadata"].get("baseline_detailed", {})
    r_base = np.array(base_det.get("span_radii", [])) / ROTOR_RADIUS
    if len(r_base) > 0:
        axes[0].plot(r_base, base_det["span_chord"], color="#9CA3AF", ls="-", lw=1.7, zorder=2, label="Baseline (NREL Phase VI)")
        axes[1].plot(r_base, base_det["span_twist_deg"], color="#9CA3AF", ls="-", lw=1.7, zorder=2, label="Baseline (NREL Phase VI)")
        axes[2].plot(r_base, base_det["span_alpha_deg"], color="#9CA3AF", ls="-", lw=1.7, zorder=2, label="Baseline (NREL Phase VI)")

    for m in GROUP1_3D:
        rows = d3["results"].get(m, [])
        if not rows:
            continue
        best_row = max(rows, key=lambda r: r["best_score"])
        det = best_row["detailed"]
        r_frac = np.array(det["span_radii"]) / ROTOR_RADIUS
        st = STYLE_MAIN_3D[m]
        axes[0].plot(r_frac, det["span_chord"], color=st["color"], ls="-", lw=st["lw"], zorder=st["zorder"], label=st["label"])
        axes[1].plot(r_frac, det["span_twist_deg"], color=st["color"], ls="-", lw=st["lw"], zorder=st["zorder"], label=st["label"])
        axes[2].plot(r_frac, det["span_alpha_deg"], color=st["color"], ls="-", lw=st["lw"], zorder=st["zorder"], label=st["label"])

    axes[0].set_title("(a) Spanwise Chord $c(r/R)$")
    axes[0].set_ylabel("Chord $c$ (m)")
    axes[1].set_title("(b) Spanwise Twist $\\beta(r/R)$")
    axes[1].set_ylabel("Twist $\\beta$ (deg)")
    axes[2].set_title("(c) Effective AoA $\\alpha(r/R)$")
    axes[2].set_ylabel("Effective AoA $\\alpha$ (deg)")
    axes[2].axhspan(6.0, 7.5, color="#00A651", alpha=0.09)
    for ax in axes:
        ax.set_xlabel("Spanwise Fraction ($r/R$)")


def _draw_3d_station_axes(axes: np.ndarray, d3: Dict[str, Any], panel_tags: List[str]):
    model_bs = BSplineBladeModel()
    x0 = model_bs.get_x0()

    blade_items = [("Baseline", "Baseline (NREL Phase VI)", "#9CA3AF", 1.7, 2, x0)]
    for m in GROUP1_3D:
        rows = d3["results"].get(m, [])
        if not rows:
            continue
        best_row = max(rows, key=lambda r: r["best_score"])
        st = STYLE_MAIN_3D[m]
        blade_items.append((m, st["label"], st["color"], st["lw"], st["zorder"], np.array(best_row["best_x"], dtype=np.float64)))

    beta_c = np.linspace(0.0, np.pi, 240)
    x_c = 0.5 * (1.0 - np.cos(beta_c))
    stage_names = ["Root", "Mid-Span", "Tip"]

    for k in range(NUM_STATIONS):
        ax = axes[k]
        for _key, lbl, col, lw, zord, x_vec in blade_items:
            st_k = model_bs.get_station_info(x_vec)[k]
            th = st_k["theta"]
            k_af = asb.KulfanAirfoil(
                upper_weights=th[0:8],
                lower_weights=th[8:16],
                leading_edge_weight=float(th[16]),
                TE_thickness=0.0,
            )
            yu = k_af.upper_coordinates(x_c)[:, 1]
            yl = k_af.lower_coordinates(x_c)[:, 1]
            ax.plot(x_c, yu, color=col, ls="-", lw=lw, zorder=zord, label=lbl if k == 0 else None)
            ax.plot(x_c, yl, color=col, ls="-", lw=lw, zorder=zord)

        ax.set_title(f"{panel_tags[k]} Station {k+1}: {stage_names[k]} ($r = {STATION_RADII[k]:.2f}$ m)")
        ax.set_xlabel("$x/c$")
        ax.set_ylabel("$y/c$")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.24, 0.26)
        ax.set_aspect("equal", adjustable="box")


def plot_3d_blade_geometry_suite(d3: Dict[str, Any]):
    """生成 3D 展向分布图、三控制站位截面图及 2×3 综合构型图（全纯色实线）。"""
    # 1. 独立展向分布图
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
    _draw_3d_spanwise_axes(axes, d3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=5, frameon=True, fontsize=8.6)
    plt.tight_layout()
    save_fig(fig, "figures/fig6a_3d_blade_spanwise")

    # 2. 独立三站位截面图 (Root / Mid-Span / Tip)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 2.95))
    _draw_3d_station_axes(axes, d3, panel_tags=["(a)", "(b)", "(c)"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=5, frameon=True, fontsize=8.6)
    plt.tight_layout()
    save_fig(fig, "figures/fig6b_3d_blade_station_sections")

    # 3. 合并 2×3 综合构型图（上排展向气动分布 + 下排三站位截面）
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.1))
    _draw_3d_spanwise_axes(axes[0], d3)
    _draw_3d_station_axes(axes[1], d3, panel_tags=["(d)", "(e)", "(f)"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=5, frameon=True, fontsize=8.8)
    plt.tight_layout()
    save_fig(fig, "figures/fig6_3d_blade_aero_combined")


def build_report(d2: Dict[str, Any], d3: Dict[str, Any], out_path: str):
    lines: List[str] = []
    lines.append("# CoLT-MALIBO 主实验对比与单组件消融完整统计报告\n")
    lines.append("> **方法全称**：**CoLT-MALIBO**（**Co**nstrained **L**LM-guided **T**rust-region **MALIBO**）。")
    lines.append("> 本文档按学术惯例严格划分为两大实验组：**Group 1（主方法对比：3 种冻结基线 vs. CoLT-MALIBO）** 与 **Group 2（单组件消融实验：CoLT-MALIBO 分别移除 LLM、Obj-TR、Con-TR）**。可行性感知信任域加速收缩（`infeas_failure_weight=2`）与动态分位数退火（$\\gamma: 1/3 \\to 0.15$）作为共享基础改进保留于各变体中。\n")

    lines.append("## 1. 实验分组与组件配置矩阵\n")
    lines.append("| 实验分组 | 方法名称 | 代理模型 | Obj-TR（仿 TuRBO 目标信任域） | Con-TR（仿 SCBO 约束代理） | LLM 物理偏移注入 | 数据来源 |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :--- |")
    lines.append("| **Group 1 基线** | **TuRBO-1** | 高斯过程 GP（Matérn-5/2） | 是 | 否 | 否 | 冻结基线数据 |")
    lines.append("| **Group 1 基线** | **MALIBO（原版）** | Meta-ResFFN + Laplace TS + 残差 GBDT | 否（全局） | 否 | 否 | 冻结基线数据 |")
    lines.append("| **Group 1 基线** | **LLM-Meta-TuRBO** | 高斯过程 GP（Matérn-5/2） | 是 | 否 | 是 | 冻结基线数据 |")
    lines.append("| **Group 1 / 2 实验组** | **CoLT-MALIBO（完整版）** | Meta-ResFFN + Laplace TS + 残差 GBDT | **是** | **是** | **是** | **完整升级版** |")
    lines.append("| **Group 2 消融** | **w/o LLM** | Meta-ResFFN + Laplace TS + 残差 GBDT | 是 | 是 | **否** | 单组件消融 |")
    lines.append("| **Group 2 消融** | **w/o Obj-TR** | Meta-ResFFN + Laplace TS + 残差 GBDT | **否（全局）** | 是 | 是 | 单组件消融 |")
    lines.append("| **Group 2 消融** | **w/o Con-TR** | Meta-ResFFN + Laplace TS + 残差 GBDT | 是 | **否** | 是 | 单组件消融 |\n")

    # ---------------- 2D ----------------
    lines.append("## 2. 2D 强约束翼型基准（NACA 0012，Re=3.5×10⁶，α=3.5°，D=18，NFE=300，10 种子）\n")
    all_2d = GROUP1_2D + ["w/o LLM", "w/o Obj-TR", "w/o Con-TR"]
    lines.append("| 分组 | 方法 | Step-15 | 最终均值 ± 标准差 (L/D) | 中位数 [最小, 最大] | 达 160 步数 | 达 200 步数 | 最终可行率 | 在线可行率 | LLM 调用 | 单种子耗时 |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in all_2d:
        rows = d2["results"][m]
        hist = np.array([r["history"][:300] for r in rows])
        sc, s15 = hist[:, -1], hist[:, 14]
        mh = np.mean(hist, axis=0)
        feas = sum(1 for r in rows if r.get("detailed", {}).get("is_feasible", True))
        pfr = [r["post_init_feasibility_rate"] for r in rows if "post_init_feasibility_rate" in r]
        pfr_s = f"{100.0 * np.mean(pfr):.1f}%" if pfr else "--"
        calls = [len(r.get("interventions", [])) for r in rows if "interventions" in r]
        c_str = f"{np.mean(calls):.1f}" if calls else "--"
        times = [r["elapsed_seconds"] for r in rows if "elapsed_seconds" in r]
        t_str = f"{np.mean(times):.1f}s" if times else "--"
        grp = "主对比" if m in GROUP1_2D else "消融"
        disp = "TuRBO-1" if m == "Vanilla-TuRBO" else ("MALIBO (Orig.)" if m == "MALIBO" else m)
        lines.append(
            f"| {grp} | **{disp}** | {np.mean(s15):.2f}±{np.std(s15):.2f} | **{np.mean(sc):.2f} ± {np.std(sc):.2f}** | "
            f"{np.median(sc):.2f} [{np.min(sc):.1f}, {np.max(sc):.1f}] | "
            f"{first_nfe_reaching(mh, 160)} | {first_nfe_reaching(mh, 200)} | "
            f"{feas}/{len(rows)} | {pfr_s} | {c_str} | {t_str} |"
        )

    lines.append("\n### 2D 配对显著性统计检验（10 组配对随机种子）\n")
    lines.append("| 组别 | 对比组 | 均值差 Δ (L/D) | 相对变化 | 胜率 | 配对 t 值 | p (t 检验) | p (Wilcoxon) | 效应量 Cohen's d |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    pairs_2d = [
        ("主对比", "CoLT-MALIBO", "MALIBO"),
        ("主对比", "CoLT-MALIBO", "Vanilla-TuRBO"),
        ("主对比", "CoLT-MALIBO", "LLM-Meta-TuRBO"),
        ("消融", "CoLT-MALIBO", "w/o LLM"),
        ("消融", "CoLT-MALIBO", "w/o Obj-TR"),
        ("消融", "CoLT-MALIBO", "w/o Con-TR"),
    ]
    for grp, a_name, b_name in pairs_2d:
        a = np.array([r["best_score"] for r in d2["results"][a_name]])
        b = np.array([r["best_score"] for r in d2["results"][b_name]])
        st = paired_stats(a, b)
        rel = st["delta"] / np.mean(b) * 100.0
        b_disp = "TuRBO-1" if b_name == "Vanilla-TuRBO" else ("MALIBO (Orig.)" if b_name == "MALIBO" else b_name)
        lines.append(
            f"| {grp} | {a_name} vs. {b_disp} | **{st['delta']:+.2f}** | {rel:+.2f}% | {st['wins']}/{st['n']} | "
            f"{st['t']:+.3f} | {st['p_t']:.5f} | {st['p_w']:.5f} | {st['cohen_d']:+.2f} |"
        )

    # ---------------- 3D ----------------
    base_p = float(d3["metadata"]["baseline_power_kW"])
    all_3d = GROUP1_3D + ["w/o LLM", "w/o Obj-TR", "w/o Con-TR"]
    lines.append(f"\n## 3. 3D NREL Phase VI 风机叶片基准（V=7.0 m/s，72 rpm，D=63，NFE=300，基准功率={base_p:.4f} kW，10 种子）\n")
    lines.append("| 分组 | 方法 | Step-15 (kW) | 最终功率均值 ± 标准差 (kW) | 中位数 [最小, 最大] | 较基准提升 | 达 6.50 步数 | 达 6.60 步数 | 在线可行率 | LLM 调用 | 单种子耗时 |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in all_3d:
        rows = d3["results"][m]
        hist = np.array([r["history"][:300] for r in rows])
        sc, s15 = hist[:, -1], hist[:, 14]
        mh = np.mean(hist, axis=0)
        pfr = [r["post_init_feasibility_rate"] for r in rows if "post_init_feasibility_rate" in r]
        pfr_s = f"{100.0 * np.mean(pfr):.1f}%" if pfr else "--"
        calls = [len(r.get("intervention_history", [])) for r in rows if "intervention_history" in r]
        c_str = f"{np.mean(calls):.1f}" if calls else "--"
        tm = np.array([r["elapsed_seconds"] for r in rows])
        gain = (np.mean(sc) - base_p) / base_p * 100.0
        grp = "主对比" if m in GROUP1_3D else "消融"
        disp = "TuRBO-1" if m == "TuRBO" else ("MALIBO (Orig.)" if m == "MALIBO" else m)
        lines.append(
            f"| {grp} | **{disp}** | {np.mean(s15):.4f}±{np.std(s15):.4f} | **{np.mean(sc):.4f} ± {np.std(sc):.4f}** | "
            f"{np.median(sc):.4f} [{np.min(sc):.3f}, {np.max(sc):.3f}] | {gain:+.2f}% | "
            f"{first_nfe_reaching(mh, 6.50)} | {first_nfe_reaching(mh, 6.60)} | "
            f"{pfr_s} | {c_str} | {np.mean(tm):.1f}s |"
        )

    lines.append("\n### 3D 配对显著性统计检验（10 组配对随机种子）\n")
    lines.append("| 组别 | 对比组 | 功率均值差 Δ (kW) | 相对变化 | 胜率 | 配对 t 值 | p (t 检验) | p (Wilcoxon) | 效应量 Cohen's d |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    pairs_3d = [
        ("主对比", "CoLT-MALIBO", "MALIBO"),
        ("主对比", "CoLT-MALIBO", "TuRBO"),
        ("主对比", "CoLT-MALIBO", "LLM-Meta-TuRBO"),
        ("消融", "CoLT-MALIBO", "w/o LLM"),
        ("消融", "CoLT-MALIBO", "w/o Obj-TR"),
        ("消融", "CoLT-MALIBO", "w/o Con-TR"),
    ]
    for grp, a_name, b_name in pairs_3d:
        a = np.array([r["best_score"] for r in d3["results"][a_name]])
        b = np.array([r["best_score"] for r in d3["results"][b_name]])
        st = paired_stats(a, b)
        rel = st["delta"] / np.mean(b) * 100.0
        b_disp = "TuRBO-1" if b_name == "TuRBO" else ("MALIBO (Orig.)" if b_name == "MALIBO" else b_name)
        lines.append(
            f"| {grp} | {a_name} vs. {b_disp} | **{st['delta']:+.4f}** | {rel:+.2f}% | {st['wins']}/{st['n']} | "
            f"{st['t']:+.3f} | {st['p_t']:.5f} | {st['p_w']:.5f} | {st['cohen_d']:+.2f} |"
        )

    # ---------------- 全程节点样本效率 ----------------
    lines.append("\n## 4. 全程样本效率对比（关键 NFE 节点 10 种子均值）\n")
    for title_c, res_c, ord_c, fmt_c in [
        ("### 2D 翼型（L/D）", d2["results"], all_2d, "{:.2f}"),
        ("### 3D 风机叶片（功率 kW）", d3["results"], all_3d, "{:.4f}"),
    ]:
        lines.append(title_c + "\n")
        lines.append("| 方法 | NFE=25 | NFE=50 | NFE=100 | NFE=150 | NFE=200 | NFE=300 | 在线可行率 (Post-Init) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m in ord_c:
            rows = res_c[m]
            mh = np.mean(np.array([r["history"][:300] for r in rows]), axis=0)
            pfr = [r["post_init_feasibility_rate"] for r in rows if "post_init_feasibility_rate" in r]
            pfr_s = f"{100.0 * np.mean(pfr):.1f}%" if pfr else "--"
            disp = "TuRBO-1" if m in ("Vanilla-TuRBO", "TuRBO") else ("MALIBO (Orig.)" if m == "MALIBO" else m)
            cols = " | ".join(fmt_c.format(mh[k - 1]) for k in [25, 50, 100, 150, 200, 300])
            lines.append(f"| **{disp}** | {cols} | {pfr_s} |")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已生成中文统计汇总报告: {out_path}")


def main():
    os.makedirs("figures", exist_ok=True)
    with open("results/ablation_2d_results.json", "r", encoding="utf-8") as f:
        d2 = json.load(f)
    with open("results/ablation_3d_results.json", "r", encoding="utf-8") as f:
        d3 = json.load(f)

    # Group 1: 主方法对比（3 基线 vs. CoLT-MALIBO）
    plot_2d_group(
        results=d2["results"],
        order=GROUP1_2D,
        styles=STYLE_MAIN_2D,
        tag_name="Main Comparison",
        prefix_conv="figures/fig1a_2d_main_convergence",
        prefix_box="figures/fig1b_2d_main_boxplot",
        prefix_comb="figures/fig1_2d_main_combined",
    )
    plot_3d_convergence_only(
        results=d3["results"],
        order=GROUP1_3D,
        styles=STYLE_MAIN_3D,
        baseline_power=float(d3["metadata"]["baseline_power_kW"]),
        title="3D NREL Phase VI Main Comparison: Convergence (Mean ± SEM, 10 Seeds)",
        out_prefix="figures/fig2_3d_main_convergence",
    )

    # Group 2: 单组件消融实验（CoLT-MALIBO vs. w/o LLM / w/o Obj-TR / w/o Con-TR）
    plot_2d_group(
        results=d2["results"],
        order=GROUP2_ABL,
        styles=STYLE_ABLATION,
        tag_name="Ablation Study",
        prefix_conv="figures/fig3a_2d_ablation_convergence",
        prefix_box="figures/fig3b_2d_ablation_boxplot",
        prefix_comb="figures/fig3_2d_ablation_combined",
    )
    plot_3d_convergence_only(
        results=d3["results"],
        order=GROUP2_ABL,
        styles=STYLE_ABLATION,
        baseline_power=float(d3["metadata"]["baseline_power_kW"]),
        title="3D NREL Phase VI Ablation Study: Convergence (Mean ± SEM, 10 Seeds)",
        out_prefix="figures/fig4_3d_ablation_convergence",
    )

    # 几何与气动构型图（Baseline + Group 1）
    plot_2d_airfoil_profiles(d2, "figures/fig5_2d_airfoil_shapes")
    plot_3d_blade_geometry_suite(d3)

    # 中文完整统计报告
    build_report(d2, d3, "results/ablation_summary_report.md")


if __name__ == "__main__":
    main()
