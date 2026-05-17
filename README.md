<div align="center">

# IoT Malware Traffic Detection with LLM-Generated Code

[![English](https://img.shields.io/badge/README-English-2ea44f?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-red?style=for-the-badge)](README.zh.md)
[![Русский](https://img.shields.io/badge/README-%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue?style=for-the-badge)](README.ru.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MLP-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis)

**A multilingual experiment on whether LLM prompts can generate reliable Python programs for IoT malware traffic classification, explainability, and adversarial robustness testing.**

</div>

---

## Overview

This repository studies a complete network-security machine-learning workflow:

1. **Binary classification** of IoT network traffic as benign or malicious.
2. **Model explainability** with Permutation Importance.
3. **Adversarial robustness testing** with gradient-based attacks against an MLP model.

The project compares hand-written baseline code with LLM-generated implementations produced from semantically equivalent prompts in **English**, **Chinese**, and **Russian**. The repository keeps the prompts, generated code, result tables, and plots so the experiment can be inspected and reproduced.

## Research Design

| Stage          | Goal                                              | Implementation                                              |
| -------------- | ------------------------------------------------- | ----------------------------------------------------------- |
| Classification | Detect benign vs. malicious network flows         | Random Forest and PyTorch MLP                               |
| Explainability | Identify features that influence model decisions  | `sklearn.inspection.permutation_importance`               |
| Attack         | Evaluate robustness of the MLP classifier         | FGSM / PGD-style adversarial perturbation experiments       |
| Comparison     | Evaluate prompt language and model-output quality | Baseline vs. LLM-generated code across EN / ZH / RU prompts |

The design follows the classification -> explanation -> attack structure used in prior IoT malware-analysis work, while replacing the original SHAP-based tree-model workflow with an MLP + Permutation Importance + gradient-attack workflow.

## Dataset

The experiment uses the Kaggle dataset [Malware Detection in Network Traffic Data](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis), which contains CTU-IoT-Malware-Capture network-flow logs.

| Item           | Description                                                                      |
| -------------- | -------------------------------------------------------------------------------- |
| Files          | 12 pipe-delimited CSV files                                                      |
| Location       | `data/CTU-IoT-Malware-Capture-*-1conn.log.labeled.csv`                         |
| Size           | Approximately 25M flow records                                                   |
| Label column   | `label`                                                                        |
| Binary mapping | `Benign` -> 0, all malicious labels -> 1                                       |
| Notes          | The local `data/` directory is ignored by Git because the raw dataset is large |

Download the data with:

```bash
python download_data.py
```

## Repository Structure

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
│   └── DS_V4pro/
└── materials/
    ├── proccess.md
    └── reference PDFs
```

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset:

```bash
python download_data.py
```

Run a simple baseline:

```bash
python baseline/baseline_simple.py
```

Run the baseline with feature importance:

```bash
python baseline/baseline_with_importance.py
```

Run the full baseline / attack workflow:

```bash
python baseline/baseline_final.py
```

Run an LLM-generated implementation, for example:

```bash
python generated_code/gpt-5.5/en_v1/prompt_v1.py
```

## Current Results Snapshot

Baseline result files are stored under `baseline/results/`.

| File                            | Contents                                     |
| ------------------------------- | -------------------------------------------- |
| `baseline_metrics.csv`        | Random Forest baseline metrics               |
| `model_comparison.csv`        | Random Forest vs. MLP classification metrics |
| `feature_importance.csv`      | Permutation Importance ranking               |
| `pgd_results_optimized.csv`   | Attack success rate by epsilon               |
| `model_comparison_attack.csv` | Metrics before and after attack experiments  |

Recorded baseline classification metrics:

| Model         | Accuracy |     F1 |
| ------------- | -------: | -----: |
| Random Forest |   0.9997 | 0.9997 |
| MLP           |   0.9990 | 0.9990 |

Generated plots are available in:

- `baseline/plots/`
- `generated_code/<model>/<language_version>/plots/`

## Prompt and Generation Matrix

| Prompt language | Prompt file                 | Generated-code examples                                               |
| --------------- | --------------------------- | --------------------------------------------------------------------- |
| English         | `prompts/en/prompt_v1.md` | `generated_code/gpt-5.5/en_v1/`, `generated_code/DS_V4pro/en_v1/` |
| Chinese         | `prompts/zh/prompt_v1.md` | `generated_code/gpt-5.5/zh_v1/`, `generated_code/DS_V4pro/zh_v1/` |
| Russian         | `prompts/ru/prompt_v1.md` | `generated_code/gpt-5.5/ru_v1/`, `generated_code/DS_V4pro/ru_v1/` |

## Output Convention

Each runnable experiment is expected to write its artifacts next to the script that produced them:

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

## Key Dependencies

| Package               | Purpose                                                       |
| --------------------- | ------------------------------------------------------------- |
| `pandas`, `numpy` | Data loading and preprocessing                                |
| `scikit-learn`      | Random Forest, metrics, preprocessing, Permutation Importance |
| `imbalanced-learn`  | Class-imbalance handling                                      |
| `matplotlib`        | Result visualization                                          |
| `kagglehub`         | Dataset download                                              |
| `torch`             | MLP model and gradient-based attacks                          |
