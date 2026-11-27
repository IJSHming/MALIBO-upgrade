"""MALIBO-upgrade 高保真水密 CFD STL 表面网格导出脚本（100% 仓库解耦）。

在 `cfd_models/` 目录生成供真实 RANS/URANS CFD 验证使用的 100% 水密 (Watertight, Closed 2-Manifold) STL 模型：
1. 2D 强约束翼型算例 (c=1.0 m, z in [0, 0.1] m, 保留精确后缘有限厚度 TE_thickness):
   - `airfoil_naca0012_baseline.stl`    : 初始 NACA 0012 基准翼型
   - `airfoil_colt_malibo_median.stl`   : CoLT-MALIBO 中位数代表性最优翼型 (与 fig5 完全一致)
2. 3D NREL Phase VI 风机叶片算例 (r in [1.2575, 5.0290] m, 0.30c 变桨轴, 闭口尾缘 t_TE=0):
   - `blade_nrel_phase6_baseline.stl`   : NREL Phase VI (S809) 原始基准叶片
   - `blade_colt_malibo_peak.stl`       : CoLT-MALIBO 10 种子峰值最优叶片 (与 fig6 完全一致)

端盖剖分严格采用逐条带 (strip) 内部三角化算法，保证无论翼型弯度或下表面 S809 反弯多强均 100% 无自交且法向一致朝外。
"""

import json
import os
from typing import Any, Dict, List, Tuple

import aerosandbox as asb
import numpy as np
import trimesh

from src.case_2d.airfoil_utils import get_naca0012_theta
from src.case_2d.evaluator import AerodynamicEvaluator
from src.case_3d.blade_geometry import (
    BASELINE_PITCH_DEG,
    BEM_NUM_SECTIONS,
    BLADE_ROOT_RADIUS,
    HUB_FLANGE_RADIUS,
    HUB_RADIUS,
    PITCH_AXIS_FRAC,
    ROTOR_RADIUS,
)
from src.case_3d.parameterizations import BSplineBladeModel


def _triangulate_airfoil_cap(n_chord: int, is_blunt_te: bool, u_idx_fn, l_idx_fn, offset: int, flip: bool) -> List[List[int]]:
    """逐条带 (strip) 精确三角化单个翼型截面端盖，绝对不出现凹多边形自交。"""
    tris: List[List[int]] = []
    # LE triangle
    tris.append([offset + u_idx_fn(0), offset + l_idx_fn(1), offset + u_idx_fn(1)])
    last_strip_end = n_chord - 1 if is_blunt_te else n_chord - 2
    for i in range(1, last_strip_end):
        u0, u1 = offset + u_idx_fn(i), offset + u_idx_fn(i + 1)
        l0, l1 = offset + l_idx_fn(i), offset + l_idx_fn(i + 1)
        tris.append([l0, l1, u1])
        tris.append([l0, u1, u0])
    if not is_blunt_te:
        tris.append([offset + l_idx_fn(n_chord - 2), offset + u_idx_fn(n_chord - 1), offset + u_idx_fn(n_chord - 2)])
    if flip:
        return [[a, c, b] for a, b, c in tris]
    return tris


def build_2d_airfoil_stl(
    theta: np.ndarray,
    chord: float = 1.0,
    span: float = 0.1,
    n_chord: int = 200,
) -> trimesh.Trimesh:
    """将 18 维 2D Kulfan CST 翼型沿 Z 轴拉伸为 100% 水密 2-Manifold STL 网格。"""
    theta = np.asarray(theta, dtype=np.float64)
    k_af = asb.KulfanAirfoil(
        upper_weights=theta[0:8],
        lower_weights=theta[8:16],
        leading_edge_weight=float(theta[16]),
        TE_thickness=float(theta[17]) if len(theta) > 17 else 0.0,
    )

    beta = np.linspace(0.0, np.pi, n_chord)
    x_c = 0.5 * (1.0 - np.cos(beta))
    yu = k_af.upper_coordinates(x_c)[:, 1]
    yl = k_af.lower_coordinates(x_c)[:, 1]
    yu[0] = 0.0
    yl[0] = 0.0
    is_blunt_te = float(yu[-1] - yl[-1]) > 1e-7

    xp = x_c * chord
    yup = yu * chord
    ylp = yl * chord

    if is_blunt_te:
        n_sec = 2 * n_chord - 1
        sec_xy = np.zeros((n_sec, 2), dtype=np.float64)
        sec_xy[0] = [xp[0], 0.0]
        sec_xy[1:n_chord] = np.column_stack([xp[1:], yup[1:]])
        sec_xy[n_chord:] = np.column_stack([xp[1:], ylp[1:]])

        def u_idx(i: int) -> int:
            return 0 if i == 0 else i

        def l_idx(i: int) -> int:
            return 0 if i == 0 else (n_chord - 1 + i)

        ccw_loop = [0] + [l_idx(i) for i in range(1, n_chord)] + [u_idx(i) for i in range(n_chord - 1, 0, -1)]
    else:
        n_sec = 2 * n_chord - 2
        sec_xy = np.zeros((n_sec, 2), dtype=np.float64)
        sec_xy[0] = [xp[0], 0.0]
        sec_xy[1:n_chord] = np.column_stack([xp[1:], yup[1:]])
        sec_xy[n_chord:] = np.column_stack([xp[1:-1], ylp[1:-1]])

        def u_idx(i: int) -> int:
            return 0 if i == 0 else i

        def l_idx(i: int) -> int:
            if i == 0:
                return 0
            if i == n_chord - 1:
                return n_chord - 1
            return n_chord - 1 + i

        ccw_loop = [0] + [l_idx(i) for i in range(1, n_chord - 1)] + [n_chord - 1] + [u_idx(i) for i in range(n_chord - 2, 0, -1)]

    v_bot = np.column_stack([sec_xy, np.zeros(n_sec)])
    v_top = np.column_stack([sec_xy, np.full(n_sec, span)])
    vertices = np.vstack([v_bot, v_top])
    off = n_sec

    faces: List[List[int]] = []
    m = len(ccw_loop)
    for k in range(m):
        a = ccw_loop[k]
        b = ccw_loop[(k + 1) % m]
        faces.append([a, b, off + b])
        faces.append([a, off + b, off + a])

    faces.extend(_triangulate_airfoil_cap(n_chord, is_blunt_te, u_idx, l_idx, offset=off, flip=False))
    faces.extend(_triangulate_airfoil_cap(n_chord, is_blunt_te, u_idx, l_idx, offset=0, flip=True))

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=True)
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:
        mesh.invert()
    assert mesh.is_watertight and mesh.is_volume, "2D 翼型 STL 未通过水密实体检验!"
    return mesh


def build_3d_blade_stl(
    x_vec: np.ndarray,
    model_bs: BSplineBladeModel,
    n_chord: int = 80,
) -> trimesh.Trimesh:
    """将 63 维 B-Spline NREL Phase VI 叶片放样为 100% 水密 2-Manifold 3D STL 实体网格。

    严格对齐 3D-wind-turbine-blade/cfd_models/ 坐标系与闭口尾缘规范：
    - Z 轴: 展向半径 r in [1.2575, 5.0290] m (31 全余弦加密截面)
    - X 轴: 弦向局部坐标 (0.30c 变桨旋转轴)
    - Y 轴: 厚度法向局部坐标
    - 闭口尖尾缘 (TE_thickness = 0.0)，根/尖端盖采用严格内部逐条带三角化。
    """
    span_data = model_bs.to_span_data(x_vec, n_stations=BEM_NUM_SECTIONS)
    r_span = span_data["r"]
    chords = span_data["chord"]
    twists_rad = span_data["twist_rad"]
    thetas = span_data["kulfan_thetas"]
    n_st = len(r_span)

    beta_c = np.linspace(0.0, np.pi, n_chord)
    x_c = 0.5 * (1.0 - np.cos(beta_c))

    # 闭口尖尾缘每个截面 2*n_chord - 2 个唯一顶点:
    # 0 = LE(0,0), 1..n_chord-1 = 上表面(1..TE), n_chord..2*n_chord-3 = 下表面(1..n_chord-2)
    n_sec = 2 * n_chord - 2

    def u_idx(i: int) -> int:
        return 0 if i == 0 else i

    def l_idx(i: int) -> int:
        if i == 0:
            return 0
        if i == n_chord - 1:
            return n_chord - 1
        return n_chord - 1 + i

    ccw_loop = [0] + [l_idx(i) for i in range(1, n_chord - 1)] + [n_chord - 1] + [u_idx(i) for i in range(n_chord - 2, 0, -1)]

    all_verts = []
    for i in range(n_st):
        c_i = float(chords[i])
        beta_i = float(twists_rad[i])
        th_i = thetas[i]
        k_af = asb.KulfanAirfoil(
            upper_weights=th_i[0:8],
            lower_weights=th_i[8:16],
            leading_edge_weight=float(th_i[16]),
            TE_thickness=0.0,
        )
        yu = k_af.upper_coordinates(x_c)[:, 1]
        yl = k_af.lower_coordinates(x_c)[:, 1]
        yu[0], yl[0] = 0.0, 0.0
        te_y = 0.5 * (yu[-1] + yl[-1])
        yu[-1], yl[-1] = te_y, te_y

        sec_x = np.empty(n_sec, dtype=np.float64)
        sec_y = np.empty(n_sec, dtype=np.float64)
        sec_x[0], sec_y[0] = x_c[0], 0.0
        sec_x[1:n_chord], sec_y[1:n_chord] = x_c[1:], yu[1:]
        sec_x[n_chord:], sec_y[n_chord:] = x_c[1:-1], yl[1:-1]

        x_loc = (sec_x - PITCH_AXIS_FRAC) * c_i
        y_loc = sec_y * c_i
        cos_b, sin_b = np.cos(beta_i), np.sin(beta_i)
        xr = x_loc * cos_b - y_loc * sin_b
        yr = x_loc * sin_b + y_loc * cos_b
        zr = np.full(n_sec, float(r_span[i]), dtype=np.float64)
        all_verts.append(np.column_stack([xr, yr, zr]))

    vertices = np.vstack(all_verts)
    faces: List[List[int]] = []

    m = len(ccw_loop)
    for i in range(n_st - 1):
        off0 = i * n_sec
        off1 = (i + 1) * n_sec
        for k in range(m):
            a = ccw_loop[k]
            b = ccw_loop[(k + 1) % m]
            faces.append([off0 + a, off0 + b, off1 + b])
            faces.append([off0 + a, off1 + b, off1 + a])

    # 根部端盖 (z = 1.2575 m, 法向 -Z) 与 叶尖端盖 (z = 5.0290 m, 法向 +Z)
    faces.extend(_triangulate_airfoil_cap(n_chord, False, u_idx, l_idx, offset=0, flip=True))
    faces.extend(_triangulate_airfoil_cap(n_chord, False, u_idx, l_idx, offset=(n_st - 1) * n_sec, flip=False))

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=True)
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:
        mesh.invert()
    assert mesh.is_watertight and mesh.is_volume, "3D 叶片 STL 未通过水密实体检验!"
    return mesh


def main():
    out_dir = "cfd_models"
    os.makedirs(out_dir, exist_ok=True)

    with open("results/ablation_2d_results.json", "r", encoding="utf-8") as f:
        d2 = json.load(f)
    with open("results/ablation_3d_results.json", "r", encoding="utf-8") as f:
        d3 = json.load(f)

    # ---------------- 1. 导出 2D 翼型水密 STL ----------------
    cond2 = d2["condition"]
    ev2 = AerodynamicEvaluator(reynolds=float(cond2["reynolds"]), alpha=float(cond2["alpha"]), max_evals=10)
    x0_2d = get_naca0012_theta()
    det0_2d = ev2.evaluate_detailed(x0_2d)

    colt_2d_runs = d2["results"]["CoLT-MALIBO"]
    scores_2d = np.array([r["best_score"] for r in colt_2d_runs], dtype=np.float64)
    med_idx_2d = int(np.argsort(scores_2d)[len(scores_2d) // 2])
    colt_2d_med = colt_2d_runs[med_idx_2d]

    items_2d = [
        (
            "airfoil_naca0012_baseline.stl",
            "基准翼型 NACA 0012",
            "N/A",
            x0_2d,
            {
                "ld": float(det0_2d["ld"]),
                "cl": float(det0_2d["cl"]),
                "cd": float(det0_2d["cd"]),
                "cm": float(det0_2d["cm"]),
                "t_max": float(det0_2d["t_max"]),
                "te_thickness": float(x0_2d[17]),
            },
        ),
        (
            "airfoil_colt_malibo_median.stl",
            "CoLT-MALIBO（2D 中位数最优）",
            str(colt_2d_med["seed"]),
            np.array(colt_2d_med["best_x"], dtype=np.float64),
            {
                "ld": float(colt_2d_med["best_score"]),
                "cl": float(colt_2d_med["detailed"]["cl"]),
                "cd": float(colt_2d_med["detailed"]["cd"]),
                "cm": float(colt_2d_med["detailed"]["cm"]),
                "t_max": float(colt_2d_med["detailed"]["t_max"]),
                "te_thickness": float(colt_2d_med["best_x"][17]),
            },
        ),
    ]

    rows_2d_md = []
    for fname, desc, seed_s, theta, info in items_2d:
        mesh = build_2d_airfoil_stl(theta, chord=1.0, span=0.1, n_chord=200)
        path = os.path.join(out_dir, fname)
        mesh.export(path)
        print(
            f"[2D {desc}] Seed={seed_s} | L/D={info['ld']:.2f} | "
            f"Watertight={mesh.is_watertight} | Vol={mesh.volume:.6f} m^3 | Area={mesh.area:.4f} m^2 -> {path}"
        )
        rows_2d_md.append(
            f"| `{fname}` | {desc} | `{seed_s}` | {info['ld']:.2f} | {info['cl']:.4f} | "
            f"{info['cd']:.5f} | {info['cm']:.4f} | {info['t_max']*100:.2f}% | {info['te_thickness']*100:.3f}% | "
            f"{mesh.is_watertight} | {mesh.volume:.6f} | {mesh.area:.4f} |"
        )

    # ---------------- 2. 导出 3D 风机叶片水密 STL ----------------
    meta3 = d3["metadata"]
    base_p = float(meta3["baseline_power_kW"])
    wind_v = float(meta3.get("wind_speed", 7.0))
    rpm_v = float(meta3.get("rpm", 72.0))
    pitch_v = float(meta3.get("pitch_angle_deg", BASELINE_PITCH_DEG))
    tsr = (rpm_v * 2.0 * np.pi / 60.0) * ROTOR_RADIUS / wind_v

    model_bs = BSplineBladeModel()
    x0_3d = model_bs.get_x0()
    colt_3d_runs = d3["results"]["CoLT-MALIBO"]
    colt_3d_peak = max(colt_3d_runs, key=lambda r: r["best_score"])

    base_thrust = float(meta3.get("baseline_detailed", {}).get("thrust_kN", 1.2134))
    max_thrust_lim = float(meta3.get("max_allowable_thrust_kN", base_thrust * 1.15))
    items_3d = [
        (
            "blade_nrel_phase6_baseline.stl",
            "基准叶片 NREL Phase VI (S809)",
            "N/A",
            x0_3d,
            base_p,
            base_thrust,
            float(meta3.get("baseline_detailed", {}).get("Cp", 0.3693)),
        ),
        (
            "blade_colt_malibo_peak.stl",
            "CoLT-MALIBO（3D 峰值最优）",
            str(colt_3d_peak["seed"]),
            np.array(colt_3d_peak["best_x"], dtype=np.float64),
            float(colt_3d_peak["best_score"]),
            float(colt_3d_peak["detailed"]["thrust_kN"]),
            float(colt_3d_peak["detailed"]["Cp"]),
        ),
    ]

    rows_3d_md = []
    base_vol = None
    for fname, desc, seed_s, x_vec, pwr, thrust, cp in items_3d:
        mesh = build_3d_blade_stl(x_vec, model_bs=model_bs, n_chord=80)
        path = os.path.join(out_dir, fname)
        mesh.export(path)
        if base_vol is None:
            base_vol = float(mesh.volume)
            dv_str = "基准"
        else:
            dv_pct = (mesh.volume - base_vol) / base_vol * 100.0
            dv_str = f"{dv_pct:+.2f}%"
        gain_str = "基准" if seed_s == "N/A" else f"{(pwr - base_p) / base_p * 100.0:+.2f}%"
        print(
            f"[3D {desc}] Seed={seed_s} | P={pwr:.4f} kW ({gain_str}) | T={thrust:.4f} kN | "
            f"Watertight={mesh.is_watertight} | Vol={mesh.volume:.6f} m^3 ({dv_str}) | Area={mesh.area:.4f} m^2 -> {path}"
        )
        rows_3d_md.append(
            f"| `{fname}` | {desc} | `{seed_s}` | {pwr:.4f} ({gain_str}) | {cp:.4f} | {thrust:.4f} | "
            f"{mesh.is_watertight} | {mesh.volume:.6f} ({dv_str}) | {mesh.area:.4f} |"
        )

    md_lines = [
        "# 高保真 CFD 验证用水密 STL 几何模型清单（MALIBO-upgrade）",
        "",
        "本目录包含 `CoLT-MALIBO`（完整升级版 MALIBO）在 2D 强约束翼型与 3D NREL Phase VI 风机叶片两个基准算例中得到的最终代表性构型，以及对应原始基准几何的 **100% 水密闭合（Watertight, Closed 2-Manifold）** STL 表面实体网格，可直接导入 OpenFOAM (`snappyHexMesh`)、ANSYS Fluent Meshing 或 STAR-CCM+ 开展高保真 RANS/URANS 气动验证。",
        "",
        "---",
        "",
        "## 1. 2D 强约束翼型算例 STL（与 `figures/fig5_2d_airfoil_shapes` 中位数构型严格一致）",
        "",
        "- **坐标系与拉伸规范**：$X \\in [0.0, 1.0]\\,\\text{m}$（单位弦长 $c = 1.0\\,\\text{m}$），$Y$ 为弯度与厚度法向（保留精确 Kulfan CST 后缘有限厚度 $t_{\\text{TE}}$），$Z \\in [0.0, 0.1]\\,\\text{m}$（展向拉伸 $b = 0.1c$ 适配 2D/对称面 RANS 网格）；上下表面各 200 余弦加密节点，端盖采用精确逐条带内部三角化。",
        f"- **基准工况**：不可压亚声速流，$Re = {float(cond2['reynolds']):.1e}$，$\\alpha = {float(cond2['alpha']):.1f}^\\circ$（海平面标准空气下 $U_\\infty \\approx 51.1\\,\\text{{m/s}}$）。",
        "",
        "| 文件名 | 几何构型 | 对应随机种子 | NeuralFoil $L/D$ | $C_L$ | $C_D$ | $C_M$ | 最大厚度 $t_{\\max}/c$ | 后缘厚度 $t_{\\text{TE}}/c$ | 水密校验 | 体积 ($\\text{m}^3$) | 表面积 ($\\text{m}^2$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        *rows_2d_md,
        "",
        "### 2D CFD 网格建议",
        "- 建议采用 SST $k$-$\\omega$（推荐耦合 $\\gamma$-$Re_\\theta$ 转捩模型以对齐 NeuralFoil/XFoil 自然转捩物理）；",
        "- 为满足 $y^+ \\le 1$，首层边界层网格厚度建议取 $y_1 \\approx 6.5 \\times 10^{-6}\\,\\text{m}$（增长率 1.15–1.20，35–45 层），远场半径 $\\ge 20c$，两侧 $z = 0$ 与 $z = 0.1\\,\\text{m}$ 设为 `symmetry` / `empty`。",
        "",
        "---",
        "",
        "## 2. 3D NREL Phase VI 风机叶片算例 STL（与 `figures/fig6_*` 峰值构型严格一致）",
        "",
        f"- **坐标系规范**（严格对齐 `3D-wind-turbine-blade/cfd_models/` 与 NREL/TP-500-29955）：",
        f"  - **Z 轴**：展向半径 $r \\in [{BLADE_ROOT_RADIUS:.4f}, {ROTOR_RADIUS:.4f}]\\,\\text{{m}}$（净气动升力段展长 $L = {ROTOR_RADIUS - BLADE_ROOT_RADIUS:.4f}\\,\\text{{m}}$，31 截面全余弦加密放样；物理轮毂法兰在 $r = {HUB_FLANGE_RADIUS:.4f}\\,\\text{{m}}$，过渡段在 $r = {HUB_RADIUS:.4f}\\text{{--}}{BLADE_ROOT_RADIUS:.4f}\\,\\text{{m}}$）；",
        f"  - **X 轴**：局部弦向，以 {int(PITCH_AXIS_FRAC*100)}% 弦长处 $(x/c = {PITCH_AXIS_FRAC:.2f})$ 为变桨扭转中心轴；",
        "  - **Y 轴**：翼型厚度法向，采用闭口尖尾缘 ($t_{\\text{TE}} = 0.0$)。",
        f"- **额定验证工况 (Sequence H)**：来流风速 $V_\\infty = {wind_v:.2f}\\,\\text{{m/s}}$，转速 $\\Omega = {rpm_v:.1f}\\,\\text{{rpm}}$（叶尖速比 $\\lambda = {tsr:.2f}$），桨距角 $\\theta_{{p,75\\%}} = {pitch_v:.3f}^\\circ$（等效叶尖 $3.0^\\circ$），轴向推力上限 $T_{{\\max}} = {max_thrust_lim:.4f}\\,\\text{{kN}}$（基准推力 $1.15\\times$），空气密度 $\\rho = 1.225\\,\\text{{kg/m}}^3$，动力黏度 $\\mu = 1.789 \\times 10^{{-5}}\\,\\text{{Pa}}\\cdot\\text{{s}}$。",
        "",
        "| 文件名 | 几何构型 | 对应随机种子 | 气动功率 (kW) | $C_p$ | 轴向推力 (kN) | 水密校验 | 实体体积 ($\\text{m}^3$) | 表面积 ($\\text{m}^2$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        *rows_3d_md,
        "",
        "### 3D CFD 网格建议",
        "- 在 $Re \\approx 8 \\times 10^5$（$V_{\\text{rel}} \\approx 26\\,\\text{m/s},\\, c \\approx 0.48\\,\\text{m}$）条件下，为使 SST $k$-$\\omega$ 满足 $y^+ \\le 1$，首层棱柱层厚度建议取 $y_1 \\approx 2.5 \\times 10^{-5}\\,\\text{m}$ ($0.025\\,\\text{mm}$)；",
        "- 其余基线叶片（`TuRBO-1`、`LLM-Meta-TuRBO`、`MALIBO` 原版）的水密 STL 网格已保存在 `3D-wind-turbine-blade/cfd_models/`，本目录 `blade_colt_malibo_peak.stl` 与其采用 100% 一致的坐标系与放样规范，可直接同框进行 CFD 对比验证。",
    ]

    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"已生成中文 CFD 模型说明文档: {readme_path}")


if __name__ == "__main__":
    main()
