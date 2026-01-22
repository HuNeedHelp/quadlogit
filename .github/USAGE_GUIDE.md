# Quadlogit / Twowaypanel 使用指南

## 项目概述

**twowaypanel** 是一个 Python 包，用于在**非线性面板数据模型中进行偏差修正估计和推断**，模型具有可加性个体和时间固定效应。

该包支持以下功能：

- **模型参数**的偏差修正估计
- **平均偏效应 (APEs)**的计算，支持连续和离散回归变量

当前版本支持四种模型类别：

- **二元logit**
- **二元probit**
- **多项logit**
- **有序logit**

这些模型可以是**静态**规范（外生回归变量）或**动态**规范（包含滞后因变量）。

---

## 安装指南

### 前置要求

- **Python 3.8+** 推荐使用
- 需要标准科学 Python 库：NumPy、SciPy、pandas、PyTorch

> **PyTorch 必需**：twowaypanel 使用自动微分 (autograd) 进行数值优化。可在 https://pytorch.org/ 按官方说明安装。
> **GPU/CUDA 不是必需的** — CPU 版本的 PyTorch 对大多数用户足够。

### 安装选项

#### 1. 从 PyPI 安装（推荐）

```bash
pip install twowaypanel
```

快速检查：

```bash
python -c "import twowaypanel; print('twowaypanel imported successfully')"
```

#### 2. 从 GitHub 安装开发版本

```bash
git clone https://github.com/zizhongyan/twowaypanel.git
cd twowaypanel
pip install -e .
```

`-e` 选项进行可编辑安装，本地代码改动会立即生效。

#### 3. 从本地副本安装

如果已下载源代码（如 ZIP 文件），在仓库根目录运行：

```bash
cd /path/to/twowaypanel
pip install -e .
```

#### 4. Jupyter/Notebook 用户

在 Jupyter 中使用 `%pip` 确保包安装到与笔记本内核相同的环境中：

```python
%cd /path/to/twowaypanel
%pip install -e .
```

验证：

```python
import twowaypanel
```

---

## 快速开始教程

### 核心概念

twowaypanel 实现了 Yan et al. (2026) 提出的基于似然的偏差修正方法，包括：

- **集成似然修正**（通过先验；"先验修正"）
- **联合似然修正**（通过惩罚；"惩罚修正"）

对于 logit 和 probit 面板，包还提供了 Fernández-Val 和 Weidner (2016) 开发的**分析偏差修正**。

### 支持的模型与方法

| 模型类别  | 规范      | 通用先验/惩罚 | 模型特定先验   | 分析修正 (FW16) |
| --------- | --------- | ------------- | -------------- | --------------- |
| 二元logit | 静态      | ✓            | ✓ (Prior SE)  | ✓              |
| 二元logit | 动态      | ✓            | ✓ (Prior DE)  | ✓              |
| Probit    | 静态/动态 | ✓            | –             | ✓              |
| 多项logit | 静态      | ✓            | ✓ (Prior SML) | –              |
| 多项logit | 动态      | ✓            | ✓ (Prior DML) | –              |
| 有序logit | 静态/动态 | ✓            | –             | –              |

### 主要入口：`twowaypanel.fit()`

大多数用户通过单一函数与包交互：

```python
twowaypanel.fit(
    Y, X=None, model=None,
    prior=None, lag=0, ac=False,
    algorithm="JML",
    X_names=None, sv=None, silent=False, ape=True, cutoff0=0,
    mcmc_iters=16000, mcmc_burnin=1000, mcmc_skipsize=2, mcmc_timer=1, 
    mcmc_sv_mle=True, mcmc_diagnosis=True,
    beta_variance=None, fe_variance=0.3, block_size=8
)
```

### 输入参数说明

#### 数据数组

**因变量 (Y)**

- 必须提供，形状为 `N × T` (通常为 2D NumPy 数组)
- `N` = 个体数量，`T` = 时间周期数

**回归变量 (X)**

- 必须提供，形状为 `N × T × K` 3D NumPy 数组
- `K` = 协变量数量
- 排序顺序：第一个索引为个体 i，第二个为时间 t，第三个为协变量 k

**变量名称 (X_names)** [可选]

- Python 列表，长度为 K，用于标记输出表格
- 如果省略，使用默认标签 `X_1, X_2, …`

#### 模型选择

```python
model="logit"    # 二元logit
model="probit"   # 二元probit
model="ologit"   # 有序logit
model="mlogit"   # 多项logit
```

#### 偏差修正：`prior` 和 `ac`

**似然修正 (prior)**

- `prior=None`：不应用修正
- `prior="Generic"`：通用修正（广泛适用于各类非线性面板）
- `prior="SE"`：静态二元logit专用
- `prior="DE"`：动态二元logit专用
- `prior="SML"`：静态多项logit专用
- `prior="DML"`：动态多项logit专用

**分析修正 (ac)**

- 默认 `False`
- 仅适用于**二元logit**和**二元probit**
- 应用 Fernández-Val 和 Weidner (2016) 的修正

#### 估计算法：`algorithm`

- `algorithm="JML"`（默认）：联合最大似然估计
- `algorithm="MCMC"`：Metropolis–Hastings 抽样（用于全后验不确定量化）

#### 动态模型参数：`lag`

- 控制 HAC 式截断级别
- 通常应与数据持久性和动态规范相关

#### 输出控制

- `silent`（默认 `False`）：设为 `True` 时不打印汇总表
- `ape`（默认 `True`）：计算平均偏效应

#### 有序logit 识别：`cutoff0`

- 设置第一个截点的固定值（默认 0）

#### 起始值：`sv`

- 为优化器提供起始值（通常默认行为足够）

#### MCMC 调参（仅当 `algorithm="MCMC"` 时）

- `mcmc_iters`：MCMC 迭代次数（默认 16000）
- `mcmc_burnin`：burn-in 迭代次数（默认 1000）
- `mcmc_skipsize`：每 n 次迭代保留一次（默认 2）
- `mcmc_timer`：进度条显示频率（默认 1）
- `mcmc_sv_mle`：是否从 MLE 起始值初始化（推荐 `True`）
- `mcmc_diagnosis`：生成收敛诊断和可视化图表
- `beta_variance`、`fe_variance`：Metropolis–Hastings 提议分布方差
- `block_size`：每块更新的参数数量（默认 8）

### 返回对象和关键属性

`twowaypanel.fit()` 返回包含参数估计、标准误和（可选）平均偏效应的结果对象。

#### 主要输出（JML 和 MCMC 都有）

- **`self.paras`**：估计的**公共参数**（如回归系数）
- **`self.se`**：公共参数的渐近**标准误**
- **`self.feparas`**：估计的**固定效应参数**（个体效应和时间效应）
- **`self.success`**：估计成功指示符（1 = 成功收敛）
- **`self.fun`**：目标函数的最终值（JML 下为最终对数似然值）
- **`self.ape`**：估计的**平均偏效应**（当 `ape=True` 时）
- **`self.apese`**：APE 估计的渐近**标准误**

#### MCMC 特定输出（仅当 `algorithm="MCMC"` 时）

- **`self.theta_mcmc`**：公共参数的 MCMC 样本
- **`self.lambda_mcmc`**：固定效应参数的 MCMC 样本
- **`self.sd`**：MCMC 样本的蒙特卡罗标准差
- **`self.accRateTheta`**：公共参数的接受率
- **`self.accRateFE`**：固定效应参数的接受率

---

## 实例演示

### 例1：静态二元logit（Angrist and Evans 1998）

#### 第一步：加载并重塑数据

```python
import numpy as np
import twowaypanel

# 加载数据
data = twowaypanel.database.angristevans98()

# 排序并重塑
data = data.sort_values(["id", "year"])
N = data["id"].nunique()
T = data["year"].nunique()

# Y: 因变量 (N x T)
Y = data["lfp"].to_numpy().reshape(N, T)

# X: 协变量 (N x T x K)
X_static = data.loc[:,['kids0_2','kids3_5','kids6_17','ln_husincome']].to_numpy().reshape(N,T,4)
X_static_names = ["Kids 0–2", "Kids 3–5", "Kids 6–17", "ln(hus inc)"]
```

> **注意**：在非线性固定效应模型中，某些个体/时期可能存在欠识别（如结果无变差）。包会自动删除此类单位并报告删除数量。

#### 第二步：联合 MLE（无偏差修正）

```python
res_mle = twowaypanel.fit(
    Y, X_static,
    model="logit", prior=None,
    algorithm="JML",
    X_names=X_static_names
)
```

**输出示例**：

```
================================================================================
---- 估计结果 ---------------------------------------------------------------
           有两路固定效应的 LOGIT 面板模型
                      无偏差修正
================================================================================
个体数量：664                        观测值：5976
时间周期：9                           对数似然：-3033.74
算法：联合MLE                   耗时（秒）：3.190
================================================================================
变量名称              系数          标准误    P>|z|    [95% 置信区间]
================================================================================
Kids 0–2      -1.17434565    0.09836036   0.000    -1.36713   -0.98156
Kids 3–5      -0.59134501    0.08622960   0.000    -0.76035   -0.42234
Kids 6–17     -0.01566283    0.06075953   0.797    -0.13474    0.10342
ln(hus inc)   -0.40458145    0.09432568   0.000    -0.58945   -0.21971
================================================================================

================================================================================
---- 平均偏效应 ---------------------------------------------------------
================================================================================
变量名称              系数          标准误    P>|z|    [95% 置信区间]
================================================================================
Kids 0–2      -0.08946279    0.03765011   0.018    -0.16326   -0.01566
Kids 3–5      -0.04504924    0.01961088   0.022    -0.08348   -0.00662
Kids 6–17     -0.00119321    0.00463185   0.797    -0.01027    0.00788
ln(hus inc)   -0.03082141    0.01526161   0.044    -0.06073   -0.00091
================================================================================
注：面板数据包含欠识别观测；
    删除 797 个（共 1461）个体，0 个（共 9）时间周期
```

### 例2：静态logit with 模型特定先验（MCMC）+ 收敛诊断

```python
beta_variance = np.array([np.sqrt(1e-1),np.sqrt(1e-2),np.sqrt(1e-2),np.sqrt(5e-5),np.sqrt(5e-7)])

res_mcmc = twowaypanel.fit(
    Y, X_static,
    model="logit", prior="SE", algorithm="MCMC",
    X_names=X_static_names,
    beta_variance=beta_variance, fe_variance=0.1,
    mcmc_iters=150000, mcmc_burnin=50000, mcmc_skipsize=20,
    mcmc_timer=1, mcmc_sv_mle=True, mcmc_diagnosis=True
)
```

#### MCMC 诊断解释

- **接受率** 20%–60% 为合理范围
- **Geweke 诊断**：比较链的早期与晚期段。大 p 值（>0.05）表示无收敛证据
- 包还生成**迹图**、**直方图**和**自相关函数（ACF）**图

### 例3：分析偏差修正（Fernández-Val and Weidner, 2016）

```python
res_fw16 = twowaypanel.fit(
    Y, X_static,
    model="logit", prior=None, ac=True,
    algorithm="JML",
    X_names=X_static_names
)
```

### 例4：带通用先验的动态 Probit

```python
X_dynamic = data.loc[:,['lag_lfp','kids0_2','kids3_5','kids6_17','ln_husincome']].to_numpy().reshape(N,T,5)
X_dynamic_names = ["LFP_t-1", "Kids 0–2", "Kids 3–5", "Kids 6–17", "ln(hus inc)"]

res_dyn = twowaypanel.fit(
    Y, X_dynamic,
    model="probit", prior="Generic", lag=1,
    algorithm="JML",
    X_names=X_dynamic_names
)
```

**`lag` 参数说明**：

- 滞后系数通常大且精确，反映因变量的持久性
- `lag=1` 控制 HAC 式截断级别，用于构造偏差修正的谱期望对象

### 例5：有序logit（模拟数据）

#### 生成人工数据

```python
# 生成动态有序logit面板（默认返回长形式数组）
FEi, FEt, FE, index_i, index_t, Y, X, Ystar, alphas0, gammas0 = twowaypanel.demo.PanelGenData(
    N=45, T=15, seed=10, model="ologit", dynamic=1
)

# 重塑为 N x T 和 N x T x K
Y = Y.reshape(45, 15)
X = X.reshape(45, 15, 2)
```

#### MLE 无修正 vs 通用先验

```python
# 无修正
res_ologit_mle = twowaypanel.fit(
    Y, X, model="ologit",
    prior=None, algorithm="JML", 
    cutoff0=-2.5,  # 识别标准化
)

# 使用通用先验
res_ologit_generic = twowaypanel.fit(
    Y, X, model="ologit",
    prior="Generic", algorithm="JML", lag=1,
    cutoff0=-2.5,
)
```

---

## API 参考

### 数据生成

#### `twowaypanel.demo.PanelGenData()`

生成非线性模型的人工面板数据集。

**参数**：

- `N`：个体数量
- `T`：时间周期数
- `seed`：随机种子（默认 "2025"）
- `model`：模型类别，选项 {"logit", "probit", "mlogit", "ologit"}（默认 "logit"）
- `dynamic`：{0, 1}，是否包含滞后因变量（默认 0）

**返回**：

- `FEi`：个体效应虚拟变量矩阵，维度 `(N·T) × N`
- `FEt`：时间效应虚拟变量矩阵，维度 `(N·T) × T`
- `FE`：合并的固定效应矩阵，维度 `(N·T) × (N+T)`
- `index_i`：长形式个体索引
- `index_t`：长形式时间索引
- `Y`：长形式结果变量
- `X`：长形式回归变量
- `Ystar`：潜在指数（当适用时）
- `alphas0`：真实个体固定效应
- `gammas0`：真实时间固定效应

**注意**：返回的对象为**长形式**（维度与 `N·T` 成比例）。在使用 `twowaypanel.fit()` 时，需重塑为：

- `Y`：`(N, T)`
- `X`：`(N, T, K)`

---

## 常见问题与建议

### 数据准备

1. **缺失值**：非线性固定效应模型对缺失值敏感。使用 `dropna()` 或填充方法前处理。
2. **尺度**：考虑对协变量进行标准化，以改进数值优化。
3. **固定效应识别**：

   - 完全不变的个体（所有时期结果相同）会被自动删除
   - 这很正常；包会报告删除的个体数量

### 偏差修正选择

| 场景                | 推荐                                                                  |
| ------------------- | --------------------------------------------------------------------- |
| 小样本（N 或 T 小） | 使用 `prior="Generic"` 或模型特定先验                               |
| 大样本（N, T 都大） | `prior=None` 可能足够；检查拟合优度                                 |
| 动态规范            | 使用 `prior="DE"`（logit）或 `prior="DML"`（多项） + `lag` 参数 |
| 不确定选择          | 从 `prior="Generic"` 开始                                           |

### MCMC 调参

- **迭代次数**：从 `mcmc_iters=50000` 开始，检查诊断
- **接受率偏低**：增加 `beta_variance` / `fe_variance`
- **接受率偏高**：减少方差值
- **收敛慢**：增加 `mcmc_iters` 或 `mcmc_burnin`

### 性能优化

1. **数据大小**：对于 N > 500，MCMC 可能较慢；考虑 `algorithm="JML"`
2. **并行化**：当前版本不支持并行处理；使用 CPU 线程优化库如 MKL/Openblas
3. **起始值**：对复杂问题，`mcmc_sv_mle=True` 通常加快收敛

---

## 参考文献

### 主要论文

Yan, Zizhong., Zhang, Zhengyu., Chen, Mingli., and Li, Jingrong. (2026), *Robust Priors in Nonlinear Panel Models with Individual and Time Effects*. *arXiv e-prints*.

### 方法参考

- Fernández-Val, I., & Weidner, M. (2016). Individual and time effects in nonlinear panel models with large N, small T. Journal of Econometrics, 192(2), 291-307.
- Graham, B. S. (2016). netrics: a Python 3.7 package for econometric analysis of networks. Available at https://github.com/bryangraham/netrics

### PyTorch 优化

- Feinman, R. (2021). Pytorch-minimize: a library for numerical optimization with autograd. Available at https://github.com/rfeinman/pytorch-minimize

---

## 项目信息

**代码维护者**：Zizhong Yan，Jinan University IESR

**Email**：helloyzz@gmail.com

**GitHub**：https://github.com/zizhongyan/twowaypanel

**版本**：0.2.1-beta

---

## 许可证

请参阅项目 LICENSE 文件。

---

**最后更新**：2026年1月
