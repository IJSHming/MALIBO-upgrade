# 高保真 CFD 验证用水密 STL 几何模型清单（MALIBO-upgrade）

本目录包含 `CoLT-MALIBO`（完整升级版 MALIBO）在 2D 强约束翼型与 3D NREL Phase VI 风机叶片两个基准算例中得到的最终代表性构型，以及对应原始基准几何的 **100% 水密闭合（Watertight, Closed 2-Manifold）** STL 表面实体网格，可直接导入 OpenFOAM (`snappyHexMesh`)、ANSYS Fluent Meshing 或 STAR-CCM+ 开展高保真 RANS/URANS 气动验证。

---

## 1. 2D 强约束翼型算例 STL（与 `figures/fig5_2d_airfoil_shapes` 中位数构型严格一致）

- **坐标系与拉伸规范**：$X \in [0.0, 1.0]\,\text{m}$（单位弦长 $c = 1.0\,\text{m}$），$Y$ 为弯度与厚度法向（保留精确 Kulfan CST 后缘有限厚度 $t_{\text{TE}}$），$Z \in [0.0, 0.1]\,\text{m}$（展向拉伸 $b = 0.1c$ 适配 2D/对称面 RANS 网格）；上下表面各 200 余弦加密节点，端盖采用精确逐条带内部三角化。
- **基准工况**：不可压亚声速流，$Re = 3.5e+06$，$\alpha = 3.5^\circ$（海平面标准空气下 $U_\infty \approx 51.1\,\text{m/s}$）。

| 文件名 | 几何构型 | 对应随机种子 | NeuralFoil $L/D$ | $C_L$ | $C_D$ | $C_M$ | 最大厚度 $t_{\max}/c$ | 后缘厚度 $t_{\text{TE}}/c$ | 水密校验 | 体积 ($\text{m}^3$) | 表面积 ($\text{m}^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `airfoil_naca0012_baseline.stl` | 基准翼型 NACA 0012 | `N/A` | 65.80 | 0.3901 | 0.00593 | 0.0005 | 12.01% | 0.255% | True | 0.008221 | 0.3686 |
| `airfoil_colt_malibo_median.stl` | CoLT-MALIBO（2D 中位数最优） | `9999` | 245.86 | 1.0403 | 0.00423 | -0.1476 | 12.09% | 0.109% | True | 0.008321 | 0.3717 |

### 2D CFD 网格建议
- 建议采用 SST $k$-$\omega$（推荐耦合 $\gamma$-$Re_\theta$ 转捩模型以对齐 NeuralFoil/XFoil 自然转捩物理）；
- 为满足 $y^+ \le 1$，首层边界层网格厚度建议取 $y_1 \approx 6.5 \times 10^{-6}\,\text{m}$（增长率 1.15–1.20，35–45 层），远场半径 $\ge 20c$，两侧 $z = 0$ 与 $z = 0.1\,\text{m}$ 设为 `symmetry` / `empty`。

---

## 2. 3D NREL Phase VI 风机叶片算例 STL（与 `figures/fig6_*` 峰值构型严格一致）

- **坐标系规范**（严格对齐 `3D-wind-turbine-blade/cfd_models/` 与 NREL/TP-500-29955）：
  - **Z 轴**：展向半径 $r \in [1.2575, 5.0290]\,\text{m}$（净气动升力段展长 $L = 3.7715\,\text{m}$，31 截面全余弦加密放样；物理轮毂法兰在 $r = 0.5083\,\text{m}$，过渡段在 $r = 0.8835\text{--}1.2575\,\text{m}$）；
  - **X 轴**：局部弦向，以 30% 弦长处 $(x/c = 0.30)$ 为变桨扭转中心轴；
  - **Y 轴**：翼型厚度法向，采用闭口尖尾缘 ($t_{\text{TE}} = 0.0$)。
- **额定验证工况 (Sequence H)**：来流风速 $V_\infty = 7.00\,\text{m/s}$，转速 $\Omega = 72.0\,\text{rpm}$（叶尖速比 $\lambda = 5.42$），桨距角 $\theta_{p,75\%} = 4.815^\circ$（等效叶尖 $3.0^\circ$），轴向推力上限 $T_{\max} = 1.3954\,\text{kN}$（基准推力 $1.15\times$），空气密度 $\rho = 1.225\,\text{kg/m}^3$，动力黏度 $\mu = 1.789 \times 10^{-5}\,\text{Pa}\cdot\text{s}$。

| 文件名 | 几何构型 | 对应随机种子 | 气动功率 (kW) | $C_p$ | 轴向推力 (kN) | 水密校验 | 实体体积 ($\text{m}^3$) | 表面积 ($\text{m}^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `blade_nrel_phase6_baseline.stl` | 基准叶片 NREL Phase VI (S809) | `N/A` | 6.0884 (基准) | 0.3647 | 1.2134 | True | 0.144600 (基准) | 4.3492 |
| `blade_colt_malibo_peak.stl` | CoLT-MALIBO（3D 峰值最优） | `3407` | 6.7183 (+10.35%) | 0.4025 | 1.3951 | True | 0.134697 (-6.85%) | 4.2441 |

### 3D CFD 网格建议
- 在 $Re \approx 8 \times 10^5$（$V_{\text{rel}} \approx 26\,\text{m/s},\, c \approx 0.48\,\text{m}$）条件下，为使 SST $k$-$\omega$ 满足 $y^+ \le 1$，首层棱柱层厚度建议取 $y_1 \approx 2.5 \times 10^{-5}\,\text{m}$ ($0.025\,\text{mm}$)；
- 其余基线叶片（`TuRBO-1`、`LLM-Meta-TuRBO`、`MALIBO` 原版）的水密 STL 网格已保存在 `3D-wind-turbine-blade/cfd_models/`，本目录 `blade_colt_malibo_peak.stl` 与其采用 100% 一致的坐标系与放样规范，可直接同框进行 CFD 对比验证。
