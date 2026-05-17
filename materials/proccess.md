# 物联网协议考查作业 —— 完成流程

## 作业目标

验证**仅通过 LLM 对话**（prompt）能否生成可用的网络恶意流量检测 Python 程序，并与手写模型对比。

采用方法：MLP 分类器 + Permutation Importance 可解释性分析 + FGSM 对抗攻击。

## 背景参考：2025 届论文

[Егоров, М. Э., et al. "Объяснения моделей машинного обучения и состязательные атаки." *International Journal of Open Information Technologies* 13.9 (2025): 50-59.](https://github.com/lava-aaa/iot_hw)

论文做了类似的事情：用 Random Forest 做 IoT 流量分类，用 SHAP 解释模型，再基于解释结果做对抗攻击。我们沿用相同的思路框架（分类 → 解释 → 攻击），但具体方法替换为 MLP + Permutation Importance + FGSM。

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
| 文件 | `data/CTU-IoT-Malware-Capture-*.csv`（12 个文件） |
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
| 7. 跑通 Baseline 攻击 | 手写 FGSM 对抗攻击 | 基准攻击成功率 |

---

## 阶段二：设计 Prompt（约 1-2 天）

| 步骤 | 做什么 | 产出 |
|---|---|---|
| 7. 写 Prompt | 中文 / 英文 / 俄文各一版，语义等价 | 三个 prompt 文件 |
| 8. 小范围试跑 | 扔给 GPT 和 Claude，看初始输出能否运行 | 初步反馈 |

### Prompt 必须覆盖的三部分

- **分类器**：MLP + Random Forest，含预处理、特征选择、GridSearchCV 调参、评估指标
- **模型解释**：sklearn 内置 `permutation_importance`，MLP 和 RF 各一份特征排序
- **对抗攻击**：FGSM 对 MLP 进行攻击，扫描不同 ε，输出攻击前后指标对比

---

## 阶段三：与 LLM 对话迭代（约 2-3 天）

对 **GPT** 和 **Claude** 分别走：发 prompt → 拿代码 → 本地运行 → 贴报错 → 修改 → 再运行，反复直到代码正常运行并输出合理指标。

---

## 阶段四：对比实验（约 1-2 天）

| 对比维度 | 具体指标 |
|---|---|
| GPT vs Claude | 分类指标、PI 特征排序一致性、FGSM 攻击成功率、代码质量 |
| 中文 vs 英文 vs 俄文 Prompt | 三种语言 prompt 产出的模型指标是否有显著差异 |
| AI 生成 vs 手写 Baseline | 分类指标差距、PI 排序一致性、FGSM 攻击成功率差距 |

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
5. GPT 对话迭代过程 + 最终指标
6. Claude 对话迭代过程 + 最终指标
7. GPT vs Claude 全维度对比
8. AI 生成 vs 手写 Baseline 对比
9. 中文 vs 英文 vs 俄文 Prompt 效果对比
10. 结论与总结

---

## 分工建议（4 人组）

| 角色 | 负责 |
|---|---|
| A — 数据 + Baseline | 下载数据、EDA、手写 MLP+RF + PI + FGSM 全链路 |
| B — GPT 对话者 | 设计 Prompt、与 GPT 迭代对话、收集各版本生成代码 |
| C — Claude 对话者 | 用相同 Prompt 与 Claude 迭代对话、收集各版本生成代码 |
| D — 实验 + 报告 | 跑全部对比实验、画图表、写报告、做 PPT |

---

## 提交清单

- [ ] GitHub 仓库（包含以下所有内容）
- [ ] Prompt（中文 + 英文 + 俄文三版，语义等价）
- [ ] 手写 Baseline（MLP+RF 分类器 + PI 解释 + FGSM 攻击）
- [ ] 至少 2 个 LLM 各生成的多版本代码
- [ ] 对比实验结果（GPT vs Claude vs Baseline）
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
├── data/
│   ├── CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv   # 原始数据文件（管道分隔）
│   ├── CTU-IoT-Malware-Capture-3-1conn.log.labeled.csv
│   ├── ...（共 12 个文件，约 25M 条记录）
│   └── download.py               # kagglehub 自动下载脚本
│
├── baseline/                    # 手写基准代码，用于和 LLM 生成代码对比
│   ├── classifier.py            #   手写 MLP + Random Forest 分类器，含预处理和评估
│   ├── permutation_importance.py #  手写 Permutation Importance 分析
│   └── fgsm_attack.py           #   手写 FGSM 对抗攻击
│
├── prompts/                     # 三个语言的 Prompt，各版本按迭代编号
│   ├── zh/                      #   中文 Prompt
│   │   ├── prompt_v1.md         #     第一版（原始版本）
│   │   ├── prompt_v2.md         #     第二版（根据迭代修改）
│   │   └── ...
│   ├── en/                      #   英文 Prompt（与中文语义等价）
│   │   └── ...
│   └── ru/                      #   俄文 Prompt（与中文语义等价）
│       └── ...
│
├── generated_code/              # LLM 生成的代码，按 LLM 和版本号组织
│   ├── gpt/                     #   ChatGPT 生成
│   │   ├── v1/                  #     第一版生成结果
│   │   │   └── program.py       #       单文件，含分类 + PI + FGSM 三部分
│   │   ├── v2/
│   │   └── ...
│   └── claude/                  #   Claude 生成
│       ├── v1/
│       ├── v2/
│       └── ...
│
├── results/                     # 对比实验结果
│   ├── baseline_metrics.csv     #   手写 Baseline 的指标记录
│   ├── gpt_metrics.csv          #   GPT 各版本的指标记录
│   ├── claude_metrics.csv       #   Claude 各版本的指标记录
│   └── comparison_charts/       #   对比图表（柱状图、折线图等）
│       ├── gpt_vs_claude_acc.png
│       ├── gpt_vs_baseline_f1.png
│       └── ...
│
├── plots/                       # 各版本生成的图表输出
│   ├── gpt/
│   │   ├── v1/                  #   GPT v1 生成的 PNG 图表
│   │   └── ...
│   └── claude/
│       └── ...
│
├── docs/                        # 文档
│   ├── report.md                #   最终结果分析报告
│   └── presentation.pptx        #   答辩 PPT
│
└── reference/                   # 参考资料
    ├── IOT_ПОВС-2026.pdf        #   课程任务说明
    └── 2025_paper.pdf           #   2025 届论文（Егоров 等）
```

### 各目录用途一句话说明

| 目录/文件 | 用途 |
|---|---|
| `README.md` | 仓库首页，写清楚"做了什么、怎么做、结果如何"，答辩时老师第一眼看这里 |
| `data/` | 存放原始数据集，只读不写 |
| `baseline/` | 人工手写的完整代码，作为衡量 LLM 能力的标尺 |
| `prompts/` | 三个语言版本的 prompt，记录每次迭代的修改历史 |
| `generated_code/` | LLM 按 prompt 生成的代码，按 LLM 和版本号分目录存放 |
| `results/` | 所有对比实验的数值结果和图表，写报告和 PPT 时直接从这里取数据 |
| `plots/` | 各版本代码运行时输出的 PNG 图表，方便回头查看不用重跑 |
| `docs/` | 最终交付物——分析报告和答辩 PPT |
| `reference/` | 课程任务书和参考论文，答辩时可能需要引用 |

## 参考

- 2025 届论文：Егоров, М. Э., et al. "Объяснения моделей машинного обучения и состязательные атаки." *International Journal of Open Information Technologies* 13.9 (2025): 50-59.
- 2025 届代码：https://github.com/lava-aaa/iot_hw / https://github.com/DarkAvery/DDoS_classifier
- sklearn Permutation Importance：https://scikit-learn.org/stable/modules/permutation_importance.html
- FGSM 论文：Goodfellow, I. J., et al. "Explaining and Harnessing Adversarial Examples." ICLR 2015.
- DocsBot DDoS Prompt 示例：https://docsbot.ai/prompts/technical/ddos-detection-application
