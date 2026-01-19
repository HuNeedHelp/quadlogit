# Copilot 指导说明 - Quadlogit

## 项目概述

**Quadlogit** 是一个用于**网络形成模型**中**四元逻辑回归**的 Python 包。它使用专门的拟差分方法从网络数据构建信息性四元组，用于估计有向网络的参数。

该项目还包括 **twowaypanel**（具有固定效应的非线性面板数据模型）的脚手架，但这仍在开发中且示例不完整。

## 架构与关键组件

### 核心数据流

1. **输入**：邻接矩阵 `G`（N×N 二进制网络）+ 可选协变量 `X`（N×N×K）
2. **四元组构造** (`quadlogit.api.quadlogit.py`)：生成所有可能的 4 元组，过滤出信息性对比
3. **标准误计算** (`quadlogit.utils.helpers.py`)：使用预加载的排列索引进行快速张量操作
4. **估计**：通过 `quadlogit.lib.ipt.logit` 对四元组级数据进行逻辑回归

### 模块结构

- **`quadlogit/api/`**：公共接口——主要 `fit()` 类协调整个工作流
- **`quadlogit/lib/ipt/`**：Graham 的 **netrics** 计量经济学方法（逻辑回归、OLS、IV 等）
- **`quadlogit/utils/`**：四元组构造、排列索引、标准误的辅助函数
- **`quadlogit/demo/`**：蒙特卡洛研究的数据生成过程（`NetworkGenDataMC`、`GenData`）

### 关键设计模式

1. **预计算索引**：网络分析需要许多 4 元组；重排/排列索引**从 `.mat` 文件预加载**（`N100_*.mat`）用于 N ≤ 100 的情况以避免昂贵的计算
   - 参考：`quadlogit.utils.helpers.loadindex()`
   - 对于 N > 100，回退到实时计算

2. **张量操作**：大量使用 NumPy 高级索引以提高速度（例如 `G[rearragement_index[:,0],rearragement_index[:,1]]`）
   - 这允许向量化四元组构造；不要将其重构为循环

3. **静默模式**：输出通过 `silent=False` 标志控制——尊重用户的详细/安静偏好

## 开发工作流

### 安装

```bash
# 开发模式安装
pip install -e .

# Jupyter 用户（确保与内核环境相同）：
%cd /path/to/quadlogit
%pip install -e .
```

### 运行示例

- [examples/examples.ipynb](examples/examples.ipynb)：演示基本工作流（不完整，使用 twowaypanel API）
- [ShiranNetwork - Final.ipynb](ShiranNetwork%20-%20Final.ipynb)：完整的网络形成示例

### 关键依赖项

- **numpy、scipy、pandas**：核心数值计算
- **torch**（在 setup.cfg 中）：可选，用于将来的面板模型扩展
- **netrics**（在 `lib/ipt/` 中供应）：Graham 的计量经济学方法

## 重要约定与陷阱

1. **数据形状要求**：
   - 邻接矩阵：`G` 必须是 (N, N) 二进制、float64
   - 协变量：`X` 为 (N, N, K) 或从 (N, N) 自动转换为 (N, N, 1)
   - 目前仅使用**第一个协变量**（`X[:,:,0]`）；多个协变量需要重构

2. **估计限制**：
   - 当前实现：**仅有向网络**和**单一协变量**
   - 无向/相互扩展存在（在 `_readme.txt` 中引用）但不在主 API 中
   - `silent=True` 抑制输出表格；`silent=False` 打印详细结果

3. **错误处理**：
   - 缺少索引会触发从 `.mat` 文件的自动加载
   - 估计失败无声地设置 `self.success = 0` 并退出
   - 无优雅恢复——如果扩展请改进错误消息

4. **文档状态**：
   - 主要文档字符串是占位符（`---to be add a helpfile---`）
   - 代码注释为英文；文档字符串在进行中
   - README 提及不存在的功能（论文链接、完整 API 参考）

## 代码质量与测试

- 未发现单元测试——验证目前通过笔记本手动进行
- 版本：0.2.1（测试版，积极开发中）
- 作者：Zizhong Yan（主要），Shiran Hu（网络扩展）

## 添加功能时

1. **维持张量操作**：不要将向量化索引重构为循环
2. **保留静默模式**：如果用户面向，向新方法添加标志
3. **首先使用 N ≤ 100 测试**：使用预加载索引进行快速迭代；N > 100 更慢
4. **更新示例**：保持 [examples/examples.ipynb](examples/examples.ipynb) 和文档字符串同步
5. **谨慎处理多协变量情况**：当前代码注释表示需要重构

## 有用的入口点

- **主工作流**：[quadlogit/api/quadlogit.py#L27](quadlogit/api/quadlogit.py#L27)（`fit.__init__`）
- **四元组逻辑**：[quadlogit/utils/helpers.py#L14](quadlogit/utils/helpers.py#L14)（`rearragement_fast`）
- **标准误**：[quadlogit/utils/helpers.py#L22](quadlogit/utils/helpers.py#L22)（`standarderror_fast`）
- **数据生成**：[quadlogit/demo/networkDGP.py#L14](quadlogit/demo/networkDGP.py#L14)（`NetworkGenDataMC`）
