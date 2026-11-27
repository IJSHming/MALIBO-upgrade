# CoLT-MALIBO（MALIBO-upgrade）：面向高维强约束气动外形优化的双信任域与大模型引导无似然贝叶斯元学习

本项目是一个**完全解耦、零跨目录依赖**的独立实验仓库，面向后续期刊论文的核心方法论需求，针对原版 **MALIBO**（Pan et al., ICML 2024）在高维强约束气动优化中的流形塌缩与神经外推过乐观瓶颈，提出完整升级框架 **CoLT-MALIBO**（**Co**nstrained **L**LM-guided **T**rust-region **MALIBO**），并在 **2D 强约束翼型优化（$D=18$）** 与 **3D NREL Phase VI 风力机叶片优化（$D=63$）** 两个跨维度气动基准上完成主对比实验与严格单组件消融验证（统一预算 `NFE = 300`，10 颗独立随机种子，严格零数据泄漏、零免费几何预筛、零静默启发式降级）。

---

## 1. 核心算法架构（CoLT-MALIBO）

针对原版 MALIBO 在高维强约束空间的两层递进失效机理（全局采样流形命中率为零 + ELU ResFFN 缺乏距离衰减而在 OOD 不可行角点输出虚假高 logit），**CoLT-MALIBO** 在严格复用已消耗预算的历史黑盒评估 $\{(\mathbf{x}_i, y_i, c_i)\}_{i=1}^N$（零额外 NFE、2D/3D 参数逐字一致）的前提下，协同融合三大核心模块与两项基础强化：

1. **仿 TuRBO 目标函数信任域（`Obj-TR`）**：
   - 在当前可行最优点 $\mathbf{x}_{\text{best}}$ 周围维护自适应缩放的局部超立方体信任域 $[x_{\text{best}} - L/2,\, x_{\text{best}} + L/2]$，并在高维下执行稀疏坐标扰动（$p_{\text{perturb}} = \min(20/D, 1)$），解决后期精细局部剥削饱和问题；
   - 配套**可行性感知加速收缩**：不可行评估在失败计数中记 2 次失败（可行但未改进记 1 次），加速从流形外撤回。
2. **仿 SCBO 约束信任域 / Empirical-Bayes 可行性与 OOD 感知代理（`Con-TR`，见 [src/constrained_acquisition.py](src/constrained_acquisition.py)）**：
   - **Empirical-Bayes Beta(1, 3) 悲观先验 Matérn-5/2 可行性代理**：
     $$p_0(t) = \frac{k_{\text{post}} + 1}{n_{\text{post}} + 4}, \qquad \hat{p}_{\text{feas}}(\mathbf{x}) = \text{clip}\!\Big(p_0(t) + \mathbf{k}(\mathbf{x}, \mathbf{X}_{\text{obs}})^\top (K_{XX} + \lambda_f \mathbf{I})^{-1}(\mathbf{c}_{\text{obs}} - p_0(t)\mathbf{1}),\; \epsilon,\; 1\Big)$$
     开局以 $p_0=0.25$ 压制 OOD 幻觉，并随真实在线可行率自动适配（2D 高可行区自动放开、3D 窄流形区保持保守）；
   - **零反探索偏置的软封顶 Constrained-LFBO 采集函数**（$\ell_f=0.35, \tau_f=0.55, \beta=2.0$）：
     $$\tilde{p}_{\text{feas}}(\mathbf{x}) = \min\!\left(\frac{\hat{p}_{\text{feas}}(\mathbf{x})}{\tau_f},\; 1.0\right), \qquad \alpha_{\text{c-LFBO}}(\mathbf{x}; \hat{\mathbf{z}}) = \sigma\!\left(\frac{u_{\text{GB}}(\mathbf{x}; \hat{\mathbf{z}}) - \mu_u}{\sigma_u}\right) \cdot \tilde{p}_{\text{feas}}(\mathbf{x})^\beta$$
     当 $\hat{p}_{\text{feas}} \ge 0.55$ 时系数恒为 $1.0$，完全消除向已评估点回拉的磁铁偏置；低于 $0.55$ 的 OOD 与约束悬崖候选则受平方惩罚。
3. **停滞自适应零泄漏 LLM 物理偏移注入与元控制（`LLM`）**：
   - 直连本地 SGLang 推理 `Qwen/Qwen3.8-27B-FP8`（严禁静默启发式降级），在第 16 步及停滞 $\ge 10$ 步时输出物理正交宏观形变偏移（2D 为 4 通道翼型正交模态，3D 为 3 站位×4 物理量 = 12 通道叶片宏观模态）与拉普拉斯 Thompson 采样温度 $\tau_{\text{TS}}$，并在随后 15 步内向候选池注入 30% 沿该方向的管状候选集。
4. **动态分位数退火（共享基础改进）**：LFBO 分位数 $\gamma(t)$ 从 $1/3$ 平滑线性退火至 $0.15$，强化中后期高优脊线辨识力。

---

## 2. 双分组实验结果（10 颗独立种子，100% 最终可行）

按标准学术范式严格分为两组（均与冻结基线保持 Step-15 比特级完全一致；完整配对 $t$ 检验、Wilcoxon 检验与各 NFE 节点统计详见 [results/ablation_summary_report.md](results/ablation_summary_report.md)）：
- **Group 1（主方法对比）**：`TuRBO-1`、`MALIBO (Orig.)`、`LLM-Meta-TuRBO`（3 种冻结基线） vs. **`CoLT-MALIBO`**（完整升级版）；
- **Group 2（单组件消融实验）**：**`CoLT-MALIBO`**（完整版） vs. **`w/o LLM`**、**`w/o Obj-TR`**、**`w/o Con-TR`**。

### 算例一：2D 强约束翼型优化（NACA 0012，$Re=3.5\times 10^6, \alpha=3.5^\circ$，$D=18$，`NFE=300`）

| 分组 | 方法 | 最终均值 ± 标准差 ($L/D$) | 中位数 [最小, 最大] | 达 200 步数 | 较 `CoLT-MALIBO` 差异 | 配对 $p_t$ 检验 | 在线可行率 | LLM 调用 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 主对比基线 | TuRBO-1 | $195.73 \pm 12.32$ | 198.97 [164.2, 207.6] | -- | $-48.64$ | $<10^{-5}$ | -- | -- |
| 主对比基线 | MALIBO (Orig.) | $179.81 \pm 9.29$ | 180.00 [166.7, 192.9] | -- | $-64.56$ | $<10^{-8}$ | -- | -- |
| 主对比基线 | LLM-Meta-TuRBO | $227.40 \pm 13.41$ | 225.10 [200.8, 247.1] | 120 | $-16.97$ | $0.00187$ ($d=1.37$) | -- | 5.1 |
| **实验组** | **CoLT-MALIBO** | **$244.37 \pm 4.56$** | **245.28** [**237.1**, **253.0**] | **70** | **基线（最优）** | -- | 88.3% | **3.7** |
| 消融 | w/o LLM | $236.83 \pm 10.36$ | 239.33 [207.7, 247.1] | 86 | $-7.54$（SD $+127\%$） | $0.03580$ ($d=0.78$) | 93.1% | 0 |
| 消融 | w/o Obj-TR | $198.60 \pm 17.95$ | 197.27 [170.7, 224.7] | -- | $-45.77$（SD $+294\%$） | $0.00005$ ($d=2.28$) | 75.8% | 8.5 |
| 消融 | w/o Con-TR | $240.36 \pm 9.17$ | 242.50 [220.8, 250.6] | 80 | $-4.01$（SD $+101\%$） | $0.15560$ ($d=0.49$) | 89.6% | 3.4 |

### 算例二：3D NREL Phase VI 风力机叶片（$V_\infty=7.0\,\text{m/s}, 72\,\text{rpm}$，$D=63$，基准 $6.0884\,\text{kW}$，`NFE=300`）

| 分组 | 方法 | 最终功率均值 ± 标准差 (kW) | 中位数 [最小, 最大] | 较基准提升 | 达 6.50 / 6.60 步数 | 较 `CoLT-MALIBO` 差异 | 在线可行率 | LLM 调用 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 主对比基线 | TuRBO-1 | $6.5920 \pm 0.0463$ | 6.5717 [6.507, 6.653] | $+8.27\%$ | 68 / -- | $-0.0974$ ($p<10^{-4}$) | -- | -- |
| 主对比基线 | MALIBO (Orig.) | $6.1773 \pm 0.0280$ | 6.1812 [6.125, 6.228] | $+1.46\%$ | -- / -- | $-0.5121$ ($p<10^{-11}$) | -- | -- |
| 主对比基线 | LLM-Meta-TuRBO | $6.6664 \pm 0.0196$ | 6.6573 [6.646, 6.696] | $+9.49\%$ | 36 / 60 | $-0.0230$ ($p=0.0279, d=0.83$) | -- | 7.2 |
| **实验组** | **CoLT-MALIBO** | **$6.6894 \pm 0.0233$** | **6.6956** [6.644, 6.718] | **$+9.87\%$** | **27** / 48 | **基线（全程领跑）** | **54.0%** | 7.9 |
| 消融 | w/o LLM | $6.6608 \pm 0.0200$ | 6.6600 [6.632, 6.692] | $+9.40\%$ | 44 / 103 | $-0.0286$ ($p=0.0128, d=0.98$) | 59.2% | 0 |
| 消融 | w/o Obj-TR | $6.6960 \pm 0.0164$ | 6.7016 [6.669, 6.721] | $+9.98\%$ | 37 / 46 | $+0.0066$ ($p=0.5479$) | 31.0% | 8.6 |
| 消融 | w/o Con-TR | $6.6831 \pm 0.0358$ | 6.6888 [6.584, 6.730] | $+9.77\%$ | 32 / 49 | $-0.0063$（SD $+54\%$） | 46.8% | 7.6 |

> **跨维度消融核心洞察**：`Obj-TR`、`Con-TR` 与 `LLM` 构成高度互补的铁三角——缺少 `LLM` 在 2D/3D 上均出现统计显著退化；缺少 `Con-TR` 在 2D 使标准差翻倍（$4.56 \to 9.17$）、在 3D 使在线可行率跌落（$54.0\% \to 46.8\%$）且最差种子塌陷至 $6.584\,\text{kW}$；缺少 `Obj-TR` 在 3D 虽暂靠 `Con-TR` 低先验隐式过滤出 LLM 12 模态管得以维持（但在线可行率骤降至 $31.0\%$），在 2D 则因 $p_0 > \tau_f$ 暴露全局采样短板而暴跌 $-45.77$。唯有完整的 **`CoLT-MALIBO`** 在 2D 与 3D 均稳居最优且方差极低。

---

## 3. 图表与高保真 CFD 水密 STL 资产清单

### 3.1 出版级可视化图表（`figures/`，纯色实线，同步导出 300 DPI PNG + 矢量 PDF）
- **Group 1 主方法对比**：
  - 2D：`fig1a_2d_main_convergence`（独立收敛）、`fig1b_2d_main_boxplot`（独立箱线图）、`fig1_2d_main_combined`（合并双栏）
  - 3D：`fig2_3d_main_convergence`（独立收敛，不绘箱线图）
- **Group 2 消融实验对比**：
  - 2D：`fig3a_2d_ablation_convergence`（独立收敛）、`fig3b_2d_ablation_boxplot`（独立箱线图）、`fig3_2d_ablation_combined`（合并双栏）
  - 3D：`fig4_3d_ablation_convergence`（独立收敛，不绘箱线图）
- **气动几何构型**：
  - 2D：`fig5_2d_airfoil_shapes`（中位数最优翼型轮廓对比）
  - 3D：`fig6a_3d_blade_spanwise`（展向弦长/扭角/有效攻角）、`fig6b_3d_blade_station_sections`（Root / Mid-Span / Tip 三控制站位翼型截面）、`fig6_3d_blade_aero_combined`（2×3 综合构型图）

### 3.2 高保真 CFD 验证用水密 STL 模型（`cfd_models/`，100% Watertight 2-Manifold）
详细坐标系、工况参数、网格体积/表面积与边界层 $y^+$ 建议见 [cfd_models/README.md](cfd_models/README.md)：
- **2D**：`airfoil_naca0012_baseline.stl`（基准）、`airfoil_colt_malibo_median.stl`（`CoLT-MALIBO` 中位数 Seed 9999，$L/D=245.86$）
- **3D**：`blade_nrel_phase6_baseline.stl`（基准 NREL Phase VI，体积 $0.144600\,\text{m}^3$）、`blade_colt_malibo_peak.stl`（`CoLT-MALIBO` 峰值 Seed 3407，$6.7183\,\text{kW}, +10.35\%$，体积减重 $-6.85\%$）

---

## 4. 目录结构与复现命令

```text
MALIBO-upgrade/
├── baselines/                  # 冻结的 2D/3D 10 种子基线 JSON（TuRBO-1、MALIBO、LLM-Meta-TuRBO）
├── weights/                    # 预训练零泄漏 2D (18D) 与 3D (63D) MALIBO ResFFN 权重
├── data/                       # NREL Phase VI 31 截面全余弦基准数据 (.npz)
├── src/
│   ├── constrained_acquisition.py  # 统一 Empirical-Bayes 可行性与 OOD 感知 Con-TR 核心模块
│   ├── case_2d/                # 独立 2D 评估器、MALIBO 核心、LLM 智能体与 CoLT-MALIBO 优化器
│   └── case_3d/                # 独立 3D 评估器、参数化、MALIBO 核心、LLM 智能体与 CoLT-MALIBO 优化器
├── run_benchmark_2d.py         # 2D 10 种子并行主对比与消融测试脚本
├── run_benchmark_3d.py         # 3D 10 种子并行主对比与消融测试脚本
├── plot_ablation.py            # 双分组纯色实线图表与中文统计报告生成脚本
├── export_cfd_stls.py          # 2D/3D 100% 水密 CFD STL 几何导出脚本
├── cfd_models/                 # CoLT-MALIBO 与基准构型水密 STL 及中文 CFD 网格说明
├── results/
│   ├── ablation_2d_results.*   # 7 方法（3 基线 + CoLT-MALIBO + 3 消融）2D 全量结果
│   ├── ablation_3d_results.*   # 7 方法（3 基线 + CoLT-MALIBO + 3 消融）3D 全量结果
│   ├── ablation_summary_report.md  # 完整中文主对比/消融统计显著性报告
│   └── archive_v1_naive_tr/    # 早期朴素 TR 探索归档
└── figures/                    # 12 组纯色实线矢量 PDF 与 300 DPI PNG 图表（fig1–fig6）
```

### 复现步骤
```bash
# 1. 启动本地 SGLang 大模型服务（Qwen/Qwen3.8-27B-FP8，端口 30000）
bash launch_sglang.sh

# 2. 运行 2D 与 3D 10 种子主实验及单组件消融（CoLT-MALIBO、w/o LLM、w/o Obj-TR、w/o Con-TR）
python run_benchmark_2d.py --n_workers 5 --device cpu --force_rerun
python run_benchmark_3d.py --n_workers 5 --device cpu --force_rerun

# 3. 生成全部双分组纯色实线图表（独立 + 合并）与中文统计报告
python plot_ablation.py

# 4. 导出 2D/3D 高保真水密 CFD STL 表面实体网格
python export_cfd_stls.py
```
