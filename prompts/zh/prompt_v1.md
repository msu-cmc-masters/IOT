# Prompt —— 中文版

你是一位资深机器学习与安全研究员。请根据以下要求，编写一份完整、可直接运行的 Python 程序。

---

## 任务概述

对网络流量数据进行**三个阶段的完整分析**：

1. **二分类**：判断每条流量记录是"正常流量（Benign）"还是"恶意软件流量（Malware）"。训练 MLP 和 Random Forest 两个模型进行对比
2. **模型可解释性分析**：使用 **Permutation Importance** 解释模型，找出哪些特征驱动了分类决策
3. **对抗攻击**：使用 **FGSM（快速梯度符号法）** 对 MLP 模型进行白盒对抗攻击，测试模型鲁棒性

> 设计思路：论文原方案用 Random Forest（树模型，无梯度）+ SHAP + 贪心黑盒攻击。本方案改用 MLP（神经网络，可求梯度）+ Permutation Importance + FGSM 白盒攻击，形成两种范式的对照。

---

## 数据集

### 下载方式

使用 kagglehub 自动下载数据集（如已缓存则直接返回路径）：

```python
import kagglehub
import os
import shutil

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 下载最新版本的数据集
path = kagglehub.dataset_download("agungpambudi/network-malware-detection-connection-analysis")

# 将 CSV 文件拷贝到 data/ 目录
for file in os.listdir(path):
    if file.endswith('.csv'):
        src = os.path.join(path, file)
        dst = os.path.join(DATA_DIR, file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"已拷贝 {file} → {DATA_DIR}/")
```

### 数据集说明

数据集来自 Kaggle，包含 **12 个管道分隔（`|`）的 CSV 文件**，为 CTU-IoT-Malware-Capture 系列网络流量记录。原始数据在 `data/` 目录下，文件名为 `CTU-IoT-Malware-Capture-*-1conn.log.labeled.csv`。

**原始数据概况：**
- 数据总量：约 25,010,000 条记录
- 列数：23 列（8 个数值型 + 15 个字符串/类别型）
- 分隔符：`|`（管道符）
- 标签列：`label`，含 `Benign`（正常）、`Malicious`（恶意）、`Malicious   DDoS`、`Malicious   PartOfAHorizontalPortScan` 等多种类型

**数值型列（8 个）：**
`ts`, `id.orig_p`, `id.resp_p`, `missed_bytes`, `orig_pkts`, `orig_ip_bytes`, `resp_pkts`, `resp_ip_bytes`

**字符串/类别列（需要编码的 15 个）：**
`uid`, `id.orig_h`, `id.resp_h`, `proto`, `service`, `duration`, `orig_bytes`, `resp_bytes`, `conn_state`, `local_orig`, `local_resp`, `history`, `tunnel_parents`, `label`, `detailed-label`

**标签分布（全部文件合计）：**
- Benign：约 8,780,000（正常流量）
- Malicious：约 7,055,000（通用恶意）
- Malicious   DDoS：约 5,778,000
- Malicious   PartOfAHorizontalPortScan：约 3,386,000
- 其他（C&C, Attack 等）：约 11,000

---

## 第一部分：分类器（Classifier）

### 1.1 数据加载与探索

- 如 `data/` 目录下无 CSV 文件，先调用 kagglehub 下载数据集到 `data/` 目录
- 读取 `data/` 下所有 CSV 文件（管道分隔符 `|`），合并为一个 DataFrame
- 打印合并后的数据形状、各列数据类型
- 检查并输出缺失值统计
- 输出 `label` 列的类别分布（Benign 与各类 Malicious 的数量和比例）
- 将标签二值化：`Benign` → 0，其余 → 1（恶意）
- 输出二值化后的类别分布

### 1.2 数据预处理

- 删除高缺失率（> 80%）的列
- 删除与分析无关的标识符列（`uid`, `id.orig_h`, `id.resp_h`, `tunnel_parents`, `detailed-label`）
- 将 `duration`, `orig_bytes`, `resp_bytes` 字符串列转换为数值（`-` 替换为 NaN，再中位数填充）
- 其余缺失值：数值列用中位数填充，类别列用众数填充
- 对类别型列（`proto`, `service`, `conn_state`, `local_orig`, `local_resp`, `history`）使用 **Label Encoding**
- 使用 IQR 方法检测极端异常值（仅对数值列），输出异常值比例，将异常值截断至上下界
- 使用 StandardScaler 对数值特征进行标准化（MLP 必须标准化，Random Forest 不需要但统一做）
- 由于数据量巨大（~25M），建议**采样**：随机抽取 200,000 条（Benign 和 Malicious 各 100,000）用于训练
- 使用 SMOTE（来自 imbalanced-learn）处理采样后仍存在的类别不均衡

### 1.3 特征选择

- 计算特征间的相关系数矩阵，识别相关系数 > 0.95 的特征对
- 从每组高度相关特征中保留一个（保留与目标列相关性更高的那个）
- 使用 SelectFromModel（基于 Random Forest）进一步筛选，保留重要性 > 均值 50% 的特征

### 1.4 模型训练

- 使用分层 train_test_split，80% 训练 / 20% 测试，random_state=42
- 训练两个模型：
  - **MLP（PyTorch 实现，使用 MPS 加速）**：
    - 使用 `torch.nn.Module` 定义多层感知机，隐藏层结构可配置
    - 使用 **MPS 后端**（`torch.device("mps")`）在 Apple Silicon GPU 上加速训练，如 MPS 不可用则回退到 CPU
    - 使用 Adam 优化器、CrossEntropyLoss 损失函数、batch_size=512
    - 手动超参数搜索（简单 for 循环即可），搜索空间：hidden_layer_sizes=[(100,), (100, 50), (200, 100)]、learning_rate=[0.001, 0.01]、num_epochs=50（配合早停 patience=5）
    - 使用训练集的 20% 作为验证集，用于早停和选择最优超参数
    - 将训练好的 PyTorch 模型封装为一个 sklearn 兼容的 wrapper 类（包含 fit/predict 方法），以便 Part 2 使用 `permutation_importance`
  - **Random Forest**：作为对照基线，使用 sklearn GridSearchCV（3 折）调参 n_estimators=[100, 200]、max_depth=[10, 20, None]，n_jobs=-1 全核并行
- 输出每个模型的最优参数

### 1.5 模型评估

- 对两个模型在测试集上分别输出：
  - Accuracy、Precision、Recall、F1-Score（macro 和 weighted）
  - 分类报告（classification_report）
  - 混淆矩阵（保存为 PNG）
  - ROC 曲线及 AUC 值（两个模型画在同一张图上对比，保存为 PNG）
- 标注哪个模型更好

---

## 第二部分：模型可解释性分析（Permutation Importance）

### 2.1 原理说明

Permutation Importance 是一种简单直观的模型可解释性方法：**随机打乱某个特征的值，观察模型准确率下降多少。下降越多，该特征越重要。** 它不依赖模型内部结构（无论是树模型还是神经网络都能用），实现仅需 sklearn 内置的 `permutation_importance` 函数。

### 2.2 对 MLP 模型计算 Permutation Importance

- 由于 PyTorch MLP 已封装为 sklearn 兼容的 wrapper（含 `predict` 方法），可直接使用 `sklearn.inspection.permutation_importance(wrapper, X_test, y_test, n_repeats=10, scoring='accuracy')`
- 分别在训练集和测试集上计算，看排序是否一致
- 输出：
  - **Permutation Importance 排序条形图**（按 importance 从高到低），保存为 PNG
  - Top-10 关键特征及其 importance 均值和标准差

### 2.3 对 Random Forest 模型计算 Permutation Importance

- 同样使用 `permutation_importance` 计算
- 输出 Top-10 关键特征排序
- 对比 MLP 和 RF 的 Top-10 关键特征是否一致（输出交集和差异）

### 2.4 特征重要性对比

- 用柱状图并排展示 MLP 和 RF 的 Top-10 特征及其重要性，保存为 PNG
- 简要分析：两个模型依赖的特征是否相同？如果不完全相同，说明什么？

---

## 第三部分：对抗攻击（FGSM）

### 3.1 原理说明

FGSM（Fast Gradient Sign Method）是最经典的白盒对抗攻击方法。它利用模型的**梯度信息**，在输入数据上沿梯度方向添加微小扰动，使模型预测结果翻转。因为 MLP 是可微的神经网络，可以直接求梯度，所以能用 FGSM；而 Random Forest 是树模型，没有梯度，无法使用 FGSM。

核心公式：**x_adv = x + ε · sign(∇_x L(x, y))**

### 3.2 攻击实现

- 由于 MLP 已在 PyTorch 中实现，FGSM 攻击可直接使用 **autograd 自动求梯度**：
  - 将恶意样本转为 `torch.tensor` 并移到 MPS 设备，设置 `requires_grad=True`
  - 前向传播得到 logits，计算 CrossEntropyLoss（target=0，即让模型误判为正常）
  - 调用 `loss.backward()` 获得输入梯度
  - 按公式 `x_adv = x + ε · sign(∇_x L)` 一步施加扰动
- 无需数值近似或有限差分法，梯度计算精确且高效
- 从测试集中取出所有恶意样本（class=1），目标是让模型将其误判为正常（class=0）
- 使用批量处理（一次前向+反向即可处理所有恶意样本），无需逐样本循环

### 3.3 攻击参数

- ε（扰动强度）：从 [0.01, 0.05, 0.1, 0.2] 四个值中做实验，找到攻击成功率高但扰动尽可能小的 ε
- 每个 ε 对所有恶意样本执行一次 FGSM 攻击（一步到位，不需要迭代）
- 输出每个 ε 对应的攻击成功率

### 3.4 攻击结果统计

- 输出不同 ε 下的攻击成功率表格和折线图（保存为 PNG）
- 选择最佳 ε（攻击成功率 > 50% 的最小 ε），输出该 ε 下的：
  - 攻击成功率 = 攻击成功的恶意样本数 / 总恶意样本数
  - 平均扰动幅度 = 修改前后样本的 L2 距离均值
- 输出攻击后模型指标：Accuracy、Precision、Recall、F1

### 3.5 攻击前后对比

- 输出攻击前后的模型指标对比表：
  - Accuracy、Precision（对 class=1）、Recall（对 class=1）、F1（对 class=1）
- 将攻击后的混淆矩阵保存为 PNG
- 输出结论：FGSM 白盒攻击使 MLP 模型性能下降了多少？

---

## 运行环境

- 使用 **uv** 创建 Python 虚拟环境并安装依赖：
  ```bash
  uv venv
  source .venv/bin/activate  # Linux/macOS
  # .venv\Scripts\activate   # Windows
  uv pip install -r requirements.txt
  ```
- 所有依赖已在仓库根目录的 `requirements.txt` 中定义，直接使用即可

## 代码规范

- 所有代码放在一个 `.py` 文件中，按"第一部分"、"第二部分"、"第三部分"顺序组织
- 使用 `if __name__ == "__main__":` 作为入口
- 使用函数封装各步骤，函数名有明确的语义
- 关键步骤输出进度信息（print），方便运行时了解进度
- 每个部分开头用 print 输出一段简短的原理说明
- 所有图表保存为 PNG 文件到 `plots/` 目录
- 使用 `argparse` 支持命令行参数：
  - `--data_path`：数据集目录路径，默认 `data/`（从该目录读取所有 CSV 文件）
  - `--test_size`：测试集比例，默认 0.2
  - `--epsilon`：FGSM 扰动强度，默认 0.05（若指定则跳过 ε 扫描，直接用该值）

---

## 期望输出示例

```
========================================
第一部分：分类器（MLP vs Random Forest）
========================================
[1.1] 合并 12 个文件完成，数据形状: (25011003, 23)
[1.1] 缺失值统计: duration=1234, orig_bytes=567, ...
[1.1] 原始类别分布: Benign=8780158, Malicious=7055007, Malicious   DDoS=5778154, ...
[1.1] 二值化后类别分布: 0(Benign)=8780158 (35.1%), 1(Malicious)=16230845 (64.9%)
[1.2] 采样后数据形状: (200000, N)
[1.2] 采样后类别分布: 0=100000, 1=100000
[1.4] MLP 最优超参数: hidden_layers=(100, 50), lr=0.001, epochs=35
[1.4] RF 最优参数: {'max_depth': 20, 'n_estimators': 200}
[1.5] MLP  → Accuracy: 0.958, F1(weighted): 0.957, AUC: 0.991
[1.5] RF   → Accuracy: 0.962, F1(weighted): 0.961, AUC: 0.994
[1.5] 最佳模型: Random Forest（略胜 MLP 0.4 个百分点）

========================================
第二部分：Permutation Importance 解释
========================================
[2.2] MLP Top-10 关键特征（Permutation Importance）:
  1. feature_15 (0.0832 ± 0.0021)
  2. feature_28 (0.0651 ± 0.0018)
  3. feature_3  (0.0523 ± 0.0015)
  ...
[2.3] RF Top-10 关键特征:
  1. feature_15 (0.0791 ± 0.0024)
  2. feature_28 (0.0612 ± 0.0019)
  3. feature_7  (0.0489 ± 0.0016)
  ...
[2.4] MLP 和 RF 的 Top-10 交集: 8/10，排序高度一致

========================================
第三部分：FGSM 白盒对抗攻击
========================================
[3.3] ε 扫描结果:
  ε=0.01 → Attack Success: 12.3%
  ε=0.05 → Attack Success: 53.8%  ← 最佳
  ε=0.10 → Attack Success: 78.2%
  ε=0.20 → Attack Success: 91.4%
[3.4] 最佳 ε=0.05，攻击成功率: 53.80% (2421/4500)
  平均 L2 扰动: 0.231
[3.5] 攻击前后对比:
         Before Attack      After Attack
  Acc    0.958              0.712
  Prec   0.942              0.561
  Recall 0.931              0.462
  F1     0.936              0.507
[3.5] 结论：FGSM 白盒攻击使 MLP 的 F1 从 0.936 降至 0.507，下降 45.8%。
  MLP 虽然有梯度可用（能用白盒攻击），但模型本身对微小扰动缺乏抵抗力。
```
