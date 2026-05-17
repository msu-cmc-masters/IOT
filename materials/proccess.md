# 物联网协议考查作业 —— 完成流程

## 作业目标

验证**仅通过 LLM 对话**（prompt）能否生成可用的网络恶意流量检测 Python 程序，并与手写 baseline 对比。

采用方法：MLP + Random Forest 分类器、Permutation Importance 可解释性分析，以及针对 MLP 的 FGSM / PGD 风格对抗攻击实验。

## 背景参考：2025 届论文

[Егоров, М. Э., et al. "Объяснения моделей машинного обучения и состязательные атаки." *International Journal of Open Information Technologies* 13.9 (2025): 50-59.](https://github.com/lava-aaa/iot_hw)

论文做了类似的事情：用 Random Forest 做 IoT 流量分类，用 SHAP 解释模型，再基于解释结果做对抗攻击。我们沿用相同的思路框架（分类 → 解释 → 攻击），但具体方法替换为 MLP / Random Forest + Permutation Importance + 梯度对抗攻击。

## 数据集

[Malware Detection in Network Traffic Data](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis)

使用 kagglehub 自动下载数据集到 `data/` 目录：

```python
import kagglehub
import os
import shutil

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 下载数据集（如已缓存则直接返回路径）
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

数据集包含 12 个管道分隔（`|`）的 CSV 文件，为 CTU-IoT-Malware-Capture 系列网络流量记录。

| 项目 | 详情 |
|---|---|
| 文件 | `data/CTU-IoT-Malware-Capture-*-1conn.log.labeled.csv`（12 个文件） |
| 数据总量 | 约 25,010,000 条 |
| 特征 | 23 列（8 个数值 + 15 个字符串/类别） |
| 标签列 | `label`：Benign（正常）与多种恶意类型 |
| 字符串列需编码 | proto, service, conn_state, history 等需 Label Encoding 或 One-Hot |
| 数值列 | ts, id.orig_p, id.resp_p, missed_bytes, orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes |

**标签分布（全部文件合计）：**

| 标签 | 数量 | 说明 |
|---|---|---|
| Benign | ~8,780,000 | 正常流量 |
| Malicious | ~7,055,000 | 恶意流量（通用） |
| Malicious\ \ DDoS | ~5,778,000 | DDoS 攻击 |
| Malicious\ \ PartOfAHorizontalPortScan | ~3,386,000 | 水平端口扫描 |
| 其他（C&C, Attack 等） | ~11,000 | 命令与控制等 |

> **二分类处理**：将 `Benign` → 0（正常），其余所有含 `Malicious` 的标签 → 1（恶意）。

## 组队

4 人一组（全班 21 人，4 组 × 4 人 + 1 组 × 5 人）

---

## 阶段一：准备（约 1 天）

| 步骤 | 做什么 | 产出 |
|---|---|---|
| 1. 组队 | 确认组员 | 人员名单 |
| 2. 创建 GitHub 仓库 | 全组成员加入，按下方目录结构建好空文件夹 | 空仓库 |
| 3. 下载数据集 | 运行 kagglehub 代码自动下载 12 个 CSV 到 `data/` | 数据集就位 |
| 4. 数据预处理 | 合并多个 CSV、编码字符串列、处理类别不均衡、标准化 | 清洗后的训练/测试集 |
| 5. 跑通 Baseline 分类器 | 手写 MLP + Random Forest 两个分类器 | 基准指标 |
| 6. 跑通 Baseline 解释 | 手写 Permutation Importance 分析 | 基准特征重要性排序 |
| 7. 跑通 Baseline 攻击 | 手写 PGD 风格对抗攻击，并保留攻击前后指标 | 基准攻击成功率与对比图 |

---

## 阶段二：设计 Prompt（约 1-2 天）

| 步骤 | 做什么 | 产出 |
|---|---|---|
| 7. 写 Prompt | 中文 / 英文 / 俄文各一版，语义等价 | 三个 prompt 文件 |
| 8. 小范围试跑 | 扔给 `gpt-5.5` 和 `DS_V4pro`，看初始输出能否运行 | 初步反馈 |

### Prompt 必须覆盖的三部分

- **分类器**：MLP + Random Forest，含预处理、特征选择、GridSearchCV 调参、评估指标
- **模型解释**：sklearn 内置 `permutation_importance`，MLP 和 RF 各一份特征排序
- **对抗攻击**：FGSM 对 MLP 进行攻击，扫描不同 ε，输出攻击前后指标对比

---

## 阶段三：与 LLM 对话迭代（约 2-3 天）

对 **gpt-5.5** 和 **DS_V4pro** 分别走：发 prompt → 拿代码 → 本地运行 → 贴报错 → 修改 → 再运行，反复直到代码正常运行并输出合理指标。

当前仓库已保留两类主要 LLM 的中 / 英 / 俄三语 `v1` 结果：

- `generated_code/gpt-5.5/{zh_v1,en_v1,ru_v1}/prompt_v1.py`
- `generated_code/DS_V4pro/{zh_v1,en_v1,ru_v1}/prompt_v1.py`

工作区中还保留了 `kimi-k2.6`、`minimax-m2.7`、`glm-5.1` 的中文版本作为扩展尝试；这些目录目前写在 `.gitignore` 中，不作为主线提交材料。

---

## 阶段四：对比实验（约 1-2 天）

| 对比维度 | 具体指标 |
|---|---|
| gpt-5.5 vs DS_V4pro | 分类指标、PI 特征排序一致性、FGSM 攻击成功率、代码质量 |
| 中文 vs 英文 vs 俄文 Prompt | 三种语言 prompt 产出的模型指标是否有显著差异 |
| AI 生成 vs 手写 Baseline | 分类指标差距、PI 排序一致性、FGSM / PGD 风格攻击结果差距 |

### 当前结果快照

- `baseline/results/model_comparison.csv`：Random Forest Accuracy / F1 约为 0.9997，MLP Accuracy / F1 约为 0.9990。
- `baseline/results/feature_importance.csv`：当前 RF Permutation Importance 的靠前特征包括 `orig_ip_bytes`、`duration`、`id.resp_p`、`orig_pkts`。
- `baseline/results/pgd_results_optimized.csv`：记录了 ε = 0.5 到 10.0 的 PGD 风格攻击扫描结果。
- `generated_code/gpt-5.5/` 与 `generated_code/DS_V4pro/`：各语言版本均已保存混淆矩阵、Permutation Importance、ROC、FGSM 成功率和 summary table 图表。

---

## 阶段五：整理与答辩（约 2-3 天）

- 写结果报告（对比表格 + 结论，附在 GitHub README）
- 做 PPT（≤ 10 页，≤ 10 分钟）
- 整理 GitHub 仓库

### PPT 建议结构

1. 封面 + 组员
2. 任务介绍（分类 → 解释 → 攻击三条链路）
3. 数据集概览
4. Prompt 设计思路（中 / 英 / 俄）
5. gpt-5.5 对话迭代过程 + 最终指标
6. DS_V4pro 对话迭代过程 + 最终指标
7. gpt-5.5 vs DS_V4pro 全维度对比
8. AI 生成 vs 手写 Baseline 对比
9. 中文 vs 英文 vs 俄文 Prompt 效果对比
10. 结论与总结

---

## 分工建议（4 人组）

| 角色 | 负责 |
|---|---|
| A — 数据 + Baseline | 下载数据、EDA、手写 MLP+RF + PI + PGD 风格攻击全链路 |
| B — gpt-5.5 对话者 | 设计 Prompt、与 gpt-5.5 迭代对话、收集中 / 英 / 俄版本生成代码 |
| C — DS_V4pro 对话者 | 用相同 Prompt 与 DS_V4pro 迭代对话、收集中 / 英 / 俄版本生成代码 |
| D — 实验 + 报告 | 跑全部对比实验、画图表、写报告、做 PPT |

---

## 提交清单

- [ ] GitHub 仓库（包含以下所有内容）
- [ ] Prompt（中文 + 英文 + 俄文三版，语义等价）
- [ ] 手写 Baseline（MLP+RF 分类器 + PI 解释 + PGD 风格攻击）
- [ ] 至少 2 个 LLM 各生成的三语版本代码（当前主线为 `gpt-5.5` 与 `DS_V4pro`）
- [ ] 对比实验结果（gpt-5.5 vs DS_V4pro vs Baseline）
- [ ] 中文 vs 英文 vs 俄文 Prompt 效果对比
- [ ] 结果分析文档
- [ ] 研讨会答辩 PPT（≤ 10 页）

---

## 仓库目录结构及说明

```
repository/
│
├── README.md                    # 仓库总说明：任务背景、组成员、结果摘要
│
├── data/                         # 原始数据文件，本地存在但被 .gitignore 忽略
│   ├── CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv
│   ├── CTU-IoT-Malware-Capture-3-1conn.log.labeled.csv
│   └── ...（共 12 个文件，约 25M 条记录）
│
├── baseline/                    # 手写基准代码，用于和 LLM 生成代码对比
│   ├── download_data.py          #   kagglehub 自动下载脚本
│   ├── baseline_simple.py        #   Random Forest 简版 baseline
│   ├── baseline_with_mlp.py      #   加入 MLP 分类器
│   ├── baseline_with_importance.py # Random Forest Permutation Importance
│   ├── baseline_full.py          #   完整流程版本
│   ├── baseline_final.py         #   当前优化版：分类 + PGD 风格攻击
│   ├── results/                  #   baseline 指标 CSV
│   └── plots/                    #   baseline 图表 PNG
│
├── prompts/                     # 三个语言的 Prompt
│   ├── zh/                      #   中文 Prompt
│   │   └── prompt_v1.md
│   ├── en/                      #   英文 Prompt（与中文语义等价）
│   │   └── prompt_v1.md
│   └── ru/                      #   俄文 Prompt（与中文语义等价）
│       └── prompt_v1.md
│
├── generated_code/              # LLM 生成的代码，按模型和语言组织
│   ├── gpt-5.5/
│   │   ├── en_v1/
│   │   │   ├── prompt_v1.py
│   │   │   └── plots/
│   │   ├── zh_v1/
│   │   └── ru_v1/
│   └── DS_V4pro/
│       ├── en_v1/
│       ├── zh_v1/
│       └── ru_v1/
│
├── materials/                   # 流程文档与课程/论文参考资料
│   ├── proccess.md
│   ├── IOT_ПОВС-2026.pdf
│   └── obyasneniya-modeley-mashinnogo-obucheniya-i-sostyazatelnye-ataki.pdf
│
└── requirements.txt             # Python 依赖
```

### 各目录用途一句话说明

| 目录/文件 | 用途 |
|---|---|
| `README.md` | 仓库首页，写清楚"做了什么、怎么做、结果如何"，答辩时老师第一眼看这里 |
| `data/` | 存放原始数据集，只读不写 |
| `baseline/` | 人工手写的完整代码，作为衡量 LLM 能力的标尺 |
| `prompts/` | 三个语言版本的 prompt，当前保留 `prompt_v1.md` |
| `generated_code/` | LLM 按 prompt 生成的代码，按 LLM 和版本号分目录存放 |
| `baseline/results/` | 手写 baseline 的数值结果，写报告和 PPT 时可直接引用 |
| `baseline/plots/` | 手写 baseline 输出的 PNG 图表 |
| `generated_code/<model>/<language_version>/plots/` | 各 LLM 生成脚本运行后输出的 PNG 图表 |
| `materials/` | 课程任务书、参考论文和本流程说明，答辩时可能需要引用 |

## 参考

- 2025 届论文：Егоров, М. Э., et al. "Объяснения моделей машинного обучения и состязательные атаки." *International Journal of Open Information Technologies* 13.9 (2025): 50-59.
- 2025 届代码：https://github.com/lava-aaa/iot_hw / https://github.com/DarkAvery/DDoS_classifier
- sklearn Permutation Importance：https://scikit-learn.org/stable/modules/permutation_importance.html
- FGSM 论文：Goodfellow, I. J., et al. "Explaining and Harnessing Adversarial Examples." ICLR 2015.
- DocsBot DDoS Prompt 示例：https://docsbot.ai/prompts/technical/ddos-detection-application
