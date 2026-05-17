<div align="center">

# 基于 LLM 生成代码的 IoT 恶意流量检测实验

[![English](https://img.shields.io/badge/README-English-2ea44f?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-red?style=for-the-badge)](README.zh.md)
[![Русский](https://img.shields.io/badge/README-%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue?style=for-the-badge)](README.ru.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MLP-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis)

**一个多语言实验：验证 LLM 能否通过中、英、俄三种 prompt 生成可运行的 IoT 恶意流量检测、模型解释和对抗攻击 Python 程序。**

</div>

---

## 项目概览

本仓库研究一个完整的网络安全机器学习流程：

1. **二分类检测**：判断 IoT 网络流量是正常流量还是恶意流量。
2. **模型可解释性分析**：使用 Permutation Importance 识别关键特征。
3. **对抗鲁棒性测试**：对 MLP 模型进行基于梯度的对抗攻击实验。

项目将人工手写 baseline 与 LLM 生成代码进行对比。LLM 生成代码来自语义等价的 **英文**、**中文**、**俄文** prompt，仓库中保留 prompt、生成代码、结果表格和图表，方便复现与答辩展示。

## 实验设计

| 阶段 | 目标                           | 实现方式                                      |
| ---- | ------------------------------ | --------------------------------------------- |
| 分类 | 区分正常与恶意网络流           | Random Forest 与 PyTorch MLP                  |
| 解释 | 找出影响模型判断的关键特征     | `sklearn.inspection.permutation_importance` |
| 攻击 | 测试 MLP 分类器鲁棒性          | FGSM / PGD 风格对抗扰动实验                   |
| 对比 | 评估 prompt 语言与生成代码质量 | Baseline vs. LLM 生成代码，覆盖 EN / ZH / RU  |

整体流程延续“分类 -> 解释 -> 攻击”的研究框架，同时将原先偏树模型和 SHAP 的路线替换为 MLP + Permutation Importance + 梯度攻击路线。

## 数据集

实验使用 Kaggle 数据集 [Malware Detection in Network Traffic Data](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis)，数据来自 CTU-IoT-Malware-Capture 网络流量日志。

| 项目       | 说明                                                          |
| ---------- | ------------------------------------------------------------- |
| 文件       | 12 个管道分隔 CSV 文件                                        |
| 位置       | `data/CTU-IoT-Malware-Capture-*-1conn.log.labeled.csv`      |
| 规模       | 约 2500 万条网络流记录                                        |
| 标签列     | `label`                                                     |
| 二分类映射 | `Benign` -> 0，所有恶意标签 -> 1                            |
| 说明       | 原始数据较大，仓库通过 `.gitignore` 忽略本地 `data/` 目录 |

下载数据：

```bash
python download_data.py
```

## 仓库结构

```text
.
├── README.md
├── README.zh.md
├── README.ru.md
├── download_data.py
├── requirements.txt
├── prompts/
│   ├── en/prompt_v1.md
│   ├── zh/prompt_v1.md
│   └── ru/prompt_v1.md
├── baseline/
│   ├── baseline_simple.py
│   ├── baseline_with_mlp.py
│   ├── baseline_with_importance.py
│   ├── baseline_full.py
│   ├── baseline_final.py
│   ├── results/
│   └── plots/
├── generated_code/
│   ├── gpt-5.5/
│   ├── DS_V4pro/
│   ├── glm-5.1/
│   ├── kimi-k2.6/
│   └── minimax-m2.7/
└── materials/
    ├── proccess.md
    └── reference PDFs
```

## 快速开始

创建环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

下载数据集：

```bash
python download_data.py
```

运行简单 baseline：

```bash
python baseline/baseline_simple.py
```

运行带特征重要性的 baseline：

```bash
python baseline/baseline_with_importance.py
```

运行完整 baseline / 攻击流程：

```bash
python baseline/baseline_final.py
```

运行某个 LLM 生成版本，例如：

```bash
python generated_code/gpt-5.5/en_v1/prompt_v1.py
```

## 当前结果快照

Baseline 结果文件位于 `baseline/results/`。

| 文件                            | 内容                              |
| ------------------------------- | --------------------------------- |
| `baseline_metrics.csv`        | Random Forest baseline 指标       |
| `model_comparison.csv`        | Random Forest 与 MLP 分类指标对比 |
| `feature_importance.csv`      | Permutation Importance 特征排序   |
| `pgd_results_optimized.csv`   | 不同 epsilon 下的攻击成功率       |
| `model_comparison_attack.csv` | 攻击前后指标对比                  |

已记录的 baseline 分类指标：

| 模型          | Accuracy |     F1 |
| ------------- | -------: | -----: |
| Random Forest |   0.9997 | 0.9997 |
| MLP           |   0.9990 | 0.9990 |

生成图表位置：

- `baseline/plots/`
- `generated_code/<model>/<language_version>/plots/`

## Prompt 与生成矩阵

| Prompt 语言 | Prompt 文件                 | 生成代码示例                                                          |
| ----------- | --------------------------- | --------------------------------------------------------------------- |
| 英文        | `prompts/en/prompt_v1.md` | `generated_code/gpt-5.5/en_v1/`, `generated_code/DS_V4pro/en_v1/` |
| 中文        | `prompts/zh/prompt_v1.md` | `generated_code/gpt-5.5/zh_v1/`, `generated_code/DS_V4pro/zh_v1/` |
| 俄文        | `prompts/ru/prompt_v1.md` | `generated_code/gpt-5.5/ru_v1/`, `generated_code/DS_V4pro/ru_v1/` |

额外的中文生成实验保存在 `generated_code/glm-5.1/`、`generated_code/kimi-k2.6/` 和 `generated_code/minimax-m2.7/`。

## 输出约定

每个可运行实验通常会把图表输出到脚本同级目录下的 `plots/`：

```text
generated_code/<model>/<language_version>/
├── prompt_v1.py
└── plots/
    ├── confusion_matrix_random_forest.png
    ├── confusion_matrix_mlp.png
    ├── permutation_importance_comparison.png
    ├── roc_curves.png
    └── summary_table.png
```

## 主要依赖

| 依赖                  | 用途                                                |
| --------------------- | --------------------------------------------------- |
| `pandas`, `numpy` | 数据读取与预处理                                    |
| `scikit-learn`      | Random Forest、指标、预处理、Permutation Importance |
| `imbalanced-learn`  | 类别不均衡处理                                      |
| `matplotlib`        | 结果可视化                                          |
| `kagglehub`         | 数据集下载                                          |
| `torch`             | MLP 模型与基于梯度的攻击                            |
