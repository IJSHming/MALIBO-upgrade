# MALIBO-Upgrade：面向高维强约束气动外形优化的可行性感知信任域与 LLM 引导无似然贝叶斯元学习

本项目是一个**完全解耦、零跨目录依赖**的独立消融实验仓库，面向后续期刊论文的核心方法论需求，围绕原版 **MALIBO**（Pan et al., ICML 2024）开展系统性升级与 $2\times 2$ 消融验证，覆盖 **2D 强约束翼型优化（$D=18$）** 与 **3D NREL Phase VI 风力机叶片优化（$D=63$）** 两个跨维度气动基准（统一预算 `NFE = 300`，10 颗独立随机种子，严格零数据泄漏、零免费几何预筛、零静默启发式降级）。

---

## 1. 核心算法创新（v2：Empirical-Bayes 可行性与 OOD 感知约束 MALIBO）

前期诊断发现，原版 MALIBO 在高维强约束空间面临两层递进瓶颈：
1. **第一层瓶颈（全局采样流形塌缩）**：在 63D 空间中全局 5,120 点均匀随机搜索命中 `0/5,120` 个可行叶片；
2. **第二层深层瓶颈（ELU ResFFN 的 OOD 边界外推过乐观）**：即使直接套上标准信任域（旧版 `v1`），由于 ELU 神经网络缺乏高斯过程核函数的“距离衰减回归”机制，在 63D 大信任域（$L=0.4\sim 0.95$）下会对远离观测区的不可行角点输出比可行域内部更高的虚假 logit（随机候选已有 `31.5%–72%` 可行时，MALIBO Top-10 候选仍有 `80%–100%` 不可行），导致开局浪费 10–15 步评估。

为此，我们在 [src/constrained_acquisition.py](src/constrained_acquisition.py) 中提出一套**完全通用、2D/3D 逐字统一、零额外 NFE 消耗**的纯黑盒数学机制（仅依赖已支付预算的历史观测 $\{(\mathbf{x}_i, y_i, c_i)\}_{i=1}^N$，$c_i \in \{0,1\}$）：

1. **Empirical-Bayes Beta(1, 3) 悲观先验 Matérn-5/2 可行性代理**：
   $$p_0(t) = \frac{k_{\text{post}} + 1}{n_{\text{post}} + 4}, \qquad \hat{p}_{\text{feas}}(\mathbf{x}) = \text{clip}\!\Big(p_0(t) + \mathbf{k}(\mathbf{x}, \mathbf{X}_{\text{obs}})^\top (K_{XX} + \lambda_f \mathbf{I})^{-1}(\mathbf{c}_{\text{obs}} - p_0(t)\mathbf{1}),\; \epsilon,\; 1\Big)$$
   开局以 $p_0=0.25$ 压制 OOD 幻觉，随后自动适配不同维度算例的真实可行率（2D 自动放开探索、3D 自动紧守流形），并在任何已观测不可行悬崖附近把 $\hat{p}_{\text{feas}} \to 0$。
2. **零反探索偏置的软封顶 Constrained-LFBO 采集函数**（仿照 SCBO，Eriksson & Poloczek 2021）：
   $$\tilde{p}_{\text{feas}}(\mathbf{x}) = \min\!\left(\frac{\hat{p}_{\text{feas}}(\mathbf{x})}{\tau_f},\; 1.0\right), \qquad \alpha_{\text{c-LFBO}}(\mathbf{x}; \hat{\mathbf{z}}) = \sigma\!\left(\frac{u_{\text{GB}}(\mathbf{x}; \hat{\mathbf{z}}) - \mu_u}{\sigma_u}\right) \cdot \tilde{p}_{\text{feas}}(\mathbf{x})^\beta$$
   其中 $\ell_f=0.35, \tau_f=0.55, \beta=2.0$（2D/3D 完全一致）。当 $\hat{p}_{\text{feas}} \ge 0.55$ 时 $\tilde{p}_{\text{feas}} \equiv 1.0$，彻底避免将搜索拉回旧点的“反探索磁铁效应”；低于 $0.55$ 的 OOD 点或约束悬崖点则受平方抑制。
3. **可行性感知信任域收缩 + 动态 $\gamma$ 分位数退火**：不可行点在信任域状态机中计 2 次失败（可行但未改进计 1 次），加速从流形外收缩；LFBO 分位数 $\gamma(t)$ 从 $1/3$ 平滑线性退火至 $0.15$，强化后期脊线局部精修。
4. **停滞自适应零泄漏 LLM 宏观偏移量注入**：调用本地 SGLang `Qwen/Qwen3.8-27B-FP8`（非思考直出 JSON，严禁静默降级），在第 16 步及停滞 $\ge 10$ 步时输出正交平滑物理形变偏移与拉普拉斯 Thompson 采样温度 $\tau_{\text{TS}}$，并在随后 15 步执行 30% 方向切片注入。

---

## 2. 核心消融实验结果（10 颗独立种子，100% 可行）

所有升级变体均复用冻结预训练 MALIBO ResFFN 权重（`weights/`）与冻结基线数据（`baselines/`），Step-15 初始状态与基线比特级完全对齐。完整配对 $t$ 检验、Wilcoxon 检验、NFE 节点曲线及 v1 vs. v2 对比详见 [results/ablation_summary_report.md](results/ablation_summary_report.md)（旧版 v1 朴素 TR 归档保存在 `results/archive_v1_naive_tr/`）。

### 算例一：2D 强约束翼型优化（NACA 0012，$Re=3.5\times 10^6, \alpha=3.5^\circ$，$D=18$，`NFE=300`）

| 方法 | 最终均值 ± 标准差 ($L/D$) | 中位数 [最小, 最大] | 达 200 步数 | 较原版 MALIBO | 较 `LLM-Meta-TuRBO` | 平均 LLM 调用 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla-TuRBO | $195.73 \pm 12.32$ | 198.97 [164.2, 207.6] | -- | $+8.9\%$ | $-31.67$ | -- |
| MALIBO（原版） | $179.81 \pm 9.29$ | 180.00 [166.7, 192.9] | -- | 基线 | $-47.59$ | -- |
| **LLM-MALIBO**（只加 LLM） | $198.60 \pm 17.95$ | 197.27 [170.7, 224.7] | -- | $+10.5\%$ ($p=0.027$) | $-28.80$（超 TuRBO-1） | 8.5 |
| **TR-MALIBO**（只加信任域） | **$236.83 \pm 10.36$** | 239.33 [207.7, 247.1] | 86 | **$+31.7\%$** ($p<10^{-5}$) | **`+9.43`（零 LLM）** | 0 |
| LLM-Meta-TuRBO | $227.40 \pm 13.41$ | 225.10 [200.8, 247.1] | 120 | $+26.5\%$ | 基线 | 5.1 |
| **LLM-TR-MALIBO**（都加） | **$244.37 \pm 4.56$** | **245.28** [**237.1**, **253.0**] | **70** | **$+35.9\%$** ($p<10^{-8}$) | **`+16.97` ($p=0.0019$, $d=1.37$)** | **3.7** |

### 算例二：3D NREL Phase VI 风力机叶片（$V_\infty=7.0\,\text{m/s}, 72\,\text{rpm}$，$D=63$，基准 $6.0884\,\text{kW}$，`NFE=300`）

| 方法 | 最终功率均值 ± 标准差 (kW) | 中位数 [最小, 最大] | 较基准提升 | 达 6.50 步数 | 达 6.60 步数 | 较 `LLM-Meta-TuRBO` | 平均 LLM 调用 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TuRBO-1 | $6.5920 \pm 0.0463$ | 6.5717 [6.507, 6.653] | $+8.27\%$ | 68 | -- | $-0.0744$ kW | -- |
| MALIBO（原版） | $6.1773 \pm 0.0280$ | 6.1812 [6.125, 6.228] | $+1.46\%$ | -- | -- | $-0.4891$ kW | -- |
| **TR-MALIBO**（只加信任域） | **$6.6608 \pm 0.0200$** | 6.6600 [6.632, 6.692] | $+9.40\%$ | 44 | 103 | $-0.0056$ kW（零 LLM，胜 TuRBO $p=0.0033$） | 0 |
| **LLM-MALIBO**（只加 LLM） | **$6.6960 \pm 0.0164$** | **6.7016** [**6.669**, **6.721**] | **$+9.98\%$** | 37 | **46** | **`+0.0296` kW ($p=0.0102$, $d=1.02$)** | 8.6 |
| **LLM-TR-MALIBO**（都加） | **$6.6894 \pm 0.0233$** | 6.6956 [6.644, 6.718] | **$+9.87\%$** | **27** | 48 | **`+0.0230` kW ($p=0.0279$, $d=0.83$)** | 7.9 |
| LLM-Meta-TuRBO | $6.6664 \pm 0.0196$ | 6.6573 [6.646, 6.696] | $+9.49\%$ | 36 | 60 | 基线 | 7.2 |

### 约束感知核心模块（v1 朴素 TR → v2 EB 可行性与 OOD 感知 c-LFBO）消融增益
- **6/6 全线正提升**：
  - 2D `LLM-TR-MALIBO`：`235.94 ± 10.95 → 244.37 ± 4.56`（`+8.43`，标准差骤降 `58.3%`，$p=0.0326$）；
  - 2D `TR-MALIBO`：`229.62 ± 12.70 → 236.83 ± 10.36`（`+7.21`，标准差降 `18.4%`）；
  - 3D `LLM-TR-MALIBO`：`6.6689 → 6.6894 kW`（`+0.0205 kW`，8/10 胜，达 6.50 kW 从 39 步缩短至 **27 步**，全程领先 `LLM-Meta-TuRBO`）；
  - 3D `LLM-MALIBO`：`6.6464 ± 0.0595 → 6.6960 ± 0.0164 kW`（`+0.0496 kW`，10/10 全胜，标准差骤降 `72.4%`，$p=0.0268$）；
  - 3D `TR-MALIBO`：`6.6575 ± 0.0291 → 6.6608 ± 0.0200 kW`（标准差降 `31.3%`，达 6.50 kW 从 60 步缩短至 **44 步**）。

---

## 3. 目录结构与复现命令

```text
MALIBO-upgrade/
├── baselines/                  # 冻结的 2D/3D 10 种子基线 JSON（TuRBO-1、MALIBO、LLM-Meta-TuRBO）
├── weights/                    # 预训练零泄漏 2D (18D) 与 3D (63D) MALIBO ResFFN 权重
├── data/                       # NREL Phase VI 31 截面全余弦基准数据 (.npz)
├── src/
│   ├── constrained_acquisition.py  # 统一 Empirical-Bayes 可行性与 OOD 感知 c-LFBO 核心模块
│   ├── case_2d/                # 独立 2D 评估器、MALIBO 核心、LLM 智能体与升级优化器
│   └── case_3d/                # 独立 3D 评估器、参数化、MALIBO 核心、LLM 智能体与升级优化器
├── run_benchmark_2d.py         # 2D 10 种子并行消融测试脚本
├── run_benchmark_3d.py         # 3D 10 种子并行消融测试脚本
├── plot_ablation.py            # 纯色实线出版级对比图与中文统计报告生成脚本
├── results/
│   ├── ablation_2d_results.*   # 最新 v2 6 方法 2D 全量结果
│   ├── ablation_3d_results.*   # 最新 v2 6 方法 3D 全量结果
│   ├── ablation_summary_report.md  # 完整中文统计检验、里程碑与 v1 vs. v2 消融报告
│   └── archive_v1_naive_tr/    # 旧版 v1 朴素 TR 全量数据归档（供论文模块消融引用）
└── figures/                    # 纯色实线矢量 PDF 与 300 DPI PNG 图表（fig1–fig5）
```

### 复现步骤
```bash
# 1. 启动本地 SGLang 大模型服务（Qwen/Qwen3.8-27B-FP8，端口 30000）
bash launch_sglang.sh

# 2. 运行 2D 与 3D 10 种子消融实验（仅运行 TR-MALIBO、LLM-MALIBO、LLM-TR-MALIBO）
python run_benchmark_2d.py --n_workers 5 --device cpu --force_rerun
python run_benchmark_3d.py --n_workers 5 --device cpu --force_rerun

# 3. 生成全套纯色实线对比图与中文统计报告
python plot_ablation.py
```
