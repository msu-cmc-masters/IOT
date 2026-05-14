# Prompt —— English Version

You are a senior machine learning engineer and security researcher. Write a complete, runnable Python program based on the following requirements.

---

## Task Overview

Perform a **three-stage complete analysis** on network traffic data:

1. **Binary Classification**: Determine whether each flow record is "Benign" or "Malware". Train both MLP and Random Forest models for comparison
2. **Model Explainability Analysis**: Use **Permutation Importance** to explain model decisions and identify which features drive classification
3. **Adversarial Attack**: Use **FGSM (Fast Gradient Sign Method)** to perform white-box adversarial attacks on the MLP model and test model robustness

> Design rationale: The original paper uses Random Forest (tree model, no gradients) + SHAP + greedy black-box attack. This approach uses MLP (neural network, differentiable) + Permutation Importance + FGSM white-box attack, forming a contrast between two paradigms.

---

## Dataset

### Download Method

Use kagglehub to automatically download the dataset (returns cached path if already downloaded):

```python
import kagglehub
import os
import shutil

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Download the latest version of the dataset
path = kagglehub.dataset_download("agungpambudi/network-malware-detection-connection-analysis")

# Copy CSV files to data/ directory
for file in os.listdir(path):
    if file.endswith('.csv'):
        src = os.path.join(path, file)
        dst = os.path.join(DATA_DIR, file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"Copied {file} -> {DATA_DIR}/")
```

### Dataset Description

The dataset comes from Kaggle, containing **12 pipe-delimited (`|`) CSV files** from the CTU-IoT-Malware-Capture series. Raw data files are in the `data/` directory, named `CTU-IoT-Malware-Capture-*-1conn.log.labeled.csv`.

**Raw data overview:**
- Total records: ~25,010,000
- Columns: 23 (8 numerical + 15 string/categorical)
- Delimiter: `|` (pipe)
- Label column: `label`, containing `Benign`, `Malicious`, `Malicious   DDoS`, `Malicious   PartOfAHorizontalPortScan`, etc.

**Numerical columns (8):**
`ts`, `id.orig_p`, `id.resp_p`, `missed_bytes`, `orig_pkts`, `orig_ip_bytes`, `resp_pkts`, `resp_ip_bytes`

**String/categorical columns (15 to be encoded):**
`uid`, `id.orig_h`, `id.resp_h`, `proto`, `service`, `duration`, `orig_bytes`, `resp_bytes`, `conn_state`, `local_orig`, `local_resp`, `history`, `tunnel_parents`, `label`, `detailed-label`

**Label distribution (all files combined):**
- Benign: ~8,780,000 (normal traffic)
- Malicious: ~7,055,000 (generic malicious)
- Malicious   DDoS: ~5,778,000
- Malicious   PartOfAHorizontalPortScan: ~3,386,000
- Others (C&C, Attack, etc.): ~11,000

---

## Part 1: Classifier

### 1.1 Data Loading & Exploration
- If no CSV files exist under `data/`, first call kagglehub to download the dataset to `data/`
- Read all CSV files under `data/` (pipe-delimited `|`), merge into a single DataFrame
- Print merged data shape and column dtypes
- Check and report missing value counts
- Print `label` column class distribution (Benign vs. various Malicious types with counts and proportions)
- Binarize labels: `Benign` → 0, everything else → 1 (malicious)
- Print binarized class distribution

### 1.2 Data Preprocessing
- Drop columns with high missing rate (> 80%)
- Drop non-predictive identifier columns (`uid`, `id.orig_h`, `id.resp_h`, `tunnel_parents`, `detailed-label`)
- Convert `duration`, `orig_bytes`, `resp_bytes` from string to numeric (replace `-` with NaN, then fill with median)
- Other missing values: numerical columns fill with median, categorical columns fill with mode
- Use **Label Encoding** for categorical columns (`proto`, `service`, `conn_state`, `local_orig`, `local_resp`, `history`)
- Use IQR method to detect extreme outliers (numerical columns only), report outlier ratio, clip to [Q1-1.5*IQR, Q3+1.5*IQR]
- Use StandardScaler on numerical features (required for MLP; RF doesn't need it but apply uniformly)
- Due to the large data size (~25M), **sample**: randomly select 200,000 records (100,000 Benign + 100,000 Malicious) for training
- Use SMOTE (from imbalanced-learn) to handle any remaining class imbalance after sampling

### 1.3 Feature Selection
- Compute feature correlation matrix, identify feature pairs with |correlation| > 0.95
- From each group of highly correlated features, keep one (the one with higher absolute correlation with target)
- Use SelectFromModel (based on Random Forest) to further select features, keep those with importance > 50% of mean importance

### 1.4 Model Training
- Use stratified train_test_split, 80% train / 20% test, random_state=42
- Train two models:
  - **MLP (PyTorch implementation, MPS-accelerated)**:
    - Define the multi-layer perceptron using `torch.nn.Module` with configurable hidden layers
    - Use **MPS backend** (`torch.device("mps")`) for GPU acceleration on Apple Silicon; fall back to CPU if MPS is unavailable
    - Use Adam optimizer, CrossEntropyLoss, batch_size=512
    - Manual hyperparameter search (a simple for-loop is sufficient): hidden_layer_sizes=[(100,), (100, 50), (200, 100)], learning_rate=[0.001, 0.01], num_epochs=50 (with early stopping patience=5)
    - Use 20% of the training set as validation set for early stopping and hyperparameter selection
    - Wrap the trained PyTorch model in an sklearn-compatible wrapper class (with fit/predict methods) so Part 2 can use `permutation_importance`
  - **Random Forest**: as control baseline, use sklearn GridSearchCV (3-fold) to tune n_estimators=[100, 200], max_depth=[10, 20, None], n_jobs=-1 for full multi-core parallelism
- Print best parameters for each model

### 1.5 Model Evaluation
- For each model on the test set, output:
  - Accuracy, Precision, Recall, F1-Score (both macro and weighted)
  - Classification report
  - Confusion matrix (saved as PNG)
  - ROC curve and AUC value (both models plotted together on the same figure for comparison, saved as PNG)
- Indicate which model is better

---

## Part 2: Model Explainability Analysis (Permutation Importance)

### 2.1 Principle

Permutation Importance is a simple and intuitive model explainability method: **randomly shuffle the values of a feature and observe how much the model's accuracy drops. The more it drops, the more important that feature is.** It does not depend on the model's internal structure (works for both tree models and neural networks), and requires only sklearn's built-in `permutation_importance` function.

### 2.2 Compute Permutation Importance for MLP
- Since the PyTorch MLP is already wrapped as an sklearn-compatible wrapper (with `predict` method), directly use `sklearn.inspection.permutation_importance(wrapper, X_test, y_test, n_repeats=10, scoring='accuracy')`
- Compute on both training and test sets to check ranking consistency
- Output:
  - **Permutation Importance ranking bar chart** (sorted by importance descending), saved as PNG
  - Top-10 key features with their importance means and standard deviations

### 2.3 Compute Permutation Importance for Random Forest
- Similarly use `permutation_importance`
- Output Top-10 key features ranking
- Compare MLP and RF Top-10 features: are they consistent? (output intersection and differences)

### 2.4 Feature Importance Comparison
- Plot side-by-side bar chart showing MLP and RF Top-10 features with their importance scores, saved as PNG
- Brief analysis: Do both models depend on the same features? If not entirely, what does that suggest?

---

## Part 3: Adversarial Attack (FGSM)

### 3.1 Principle

FGSM (Fast Gradient Sign Method) is the most classic white-box adversarial attack method. It leverages the model's **gradient information** to add a tiny perturbation along the gradient direction on the input data, flipping the model's prediction. Because MLP is a differentiable neural network, we can compute gradients and use FGSM; Random Forest, being a tree model without gradients, cannot be attacked with FGSM.

Core formula: **x_adv = x + ε · sign(∇_x L(x, y))**

### 3.2 Attack Implementation
- Since the MLP is already implemented in PyTorch, FGSM can directly use **autograd** for gradient computation:
  - Convert malware samples to `torch.tensor` on the MPS device, set `requires_grad=True`
  - Forward pass to get logits, compute CrossEntropyLoss (target=0, i.e., trick the model into predicting benign)
  - Call `loss.backward()` to obtain input gradients
  - Apply perturbation in one step: `x_adv = x + ε · sign(∇_x L)`
- No numerical approximation or finite differences needed — gradient computation is exact and efficient
- Take all malware samples (class=1) from the test set; the goal is to make the model misclassify them as benign (class=0)
- Use batch processing (one forward+backward pass handles all malware samples at once) — no per-sample loop needed

### 3.3 Attack Parameters
- ε (perturbation strength): experiment with [0.01, 0.05, 0.1, 0.2] to find an ε that gives high attack success rate while keeping perturbations small
- For each ε, perform one FGSM attack on all malware samples (single step, no iteration needed)
- Report attack success rate for each ε

### 3.4 Attack Results Statistics
- Output a table and line chart of attack success rate vs ε (saved as PNG)
- Select the best ε (the smallest ε with attack success rate > 50%), and report:
  - Attack success rate = successful samples / total malware samples
  - Average perturbation magnitude = mean L2 distance between original and adversarial samples
- Output post-attack model metrics: Accuracy, Precision, Recall, F1

### 3.5 Before-vs-After Comparison
- Output a comparison table of model metrics before and after the attack:
  - Accuracy, Precision (for class=1), Recall (for class=1), F1 (for class=1)
- Save the post-attack confusion matrix as PNG
- Conclusion: by how much did the FGSM white-box attack degrade MLP model performance?

---

## Runtime Environment

- Use **uv** to create a Python virtual environment and install dependencies:
  ```bash
  uv venv
  source .venv/bin/activate  # Linux/macOS
  # .venv\Scripts\activate   # Windows
  uv pip install -r requirements.txt
  ```
- All dependencies are defined in the repository root's `requirements.txt`, use it directly

## Code Standards

- All code in a single `.py` file, organized as "Part 1", "Part 2", "Part 3"
- Use `if __name__ == "__main__":` as entry point
- Use functions to encapsulate each step, with semantically clear function names
- Print progress information at key steps so users can track runtime progress
- At the start of each part, print a brief explanation of the method's principle
- Save all plots as PNG files in a `plots/` directory
- Use `argparse` to support command-line arguments:
  - `--data_path`: dataset directory path, default `data/` (reads all CSV files from this directory)
  - `--test_size`: test set proportion, default 0.2
  - `--epsilon`: FGSM perturbation strength, default 0.05 (if specified, skip ε sweep and use this value directly)

---

## Expected Output Example

```
========================================
Part 1: Classifier (MLP vs Random Forest)
========================================
[1.1] Merged 12 files, data shape: (25011003, 23)
[1.1] Missing values: duration=1234, orig_bytes=567, ...
[1.1] Original class distribution: Benign=8780158, Malicious=7055007, Malicious   DDoS=5778154, ...
[1.1] Binarized class distribution: 0(Benign)=8780158 (35.1%), 1(Malicious)=16230845 (64.9%)
[1.2] After sampling, data shape: (200000, N)
[1.2] Sampled class distribution: 0=100000, 1=100000
[1.4] MLP best hyperparams: hidden_layers=(100, 50), lr=0.001, epochs=35
[1.4] RF best params: {'max_depth': 20, 'n_estimators': 200}
[1.5] MLP → Accuracy: 0.958, F1(weighted): 0.957, AUC: 0.991
[1.5] RF  → Accuracy: 0.962, F1(weighted): 0.961, AUC: 0.994
[1.5] Best model: Random Forest (beats MLP by 0.4 percentage points)

========================================
Part 2: Permutation Importance Explanation
========================================
[2.2] MLP Top-10 key features (Permutation Importance):
  1. feature_15 (0.0832 ± 0.0021)
  2. feature_28 (0.0651 ± 0.0018)
  3. feature_3  (0.0523 ± 0.0015)
  ...
[2.3] RF Top-10 key features:
  1. feature_15 (0.0791 ± 0.0024)
  2. feature_28 (0.0612 ± 0.0019)
  3. feature_7  (0.0489 ± 0.0016)
  ...
[2.4] MLP and RF Top-10 intersection: 8/10, rankings highly consistent

========================================
Part 3: FGSM White-Box Adversarial Attack
========================================
[3.3] ε sweep results:
  ε=0.01 → Attack Success: 12.3%
  ε=0.05 → Attack Success: 53.8%  ← best
  ε=0.10 → Attack Success: 78.2%
  ε=0.20 → Attack Success: 91.4%
[3.4] Best ε=0.05, Attack Success: 53.80% (2421/4500)
  Average L2 perturbation: 0.231
[3.5] Before-vs-after comparison:
         Before Attack      After Attack
  Acc    0.958              0.712
  Prec   0.942              0.561
  Recall 0.931              0.462
  F1     0.936              0.507
[3.5] Conclusion: FGSM white-box attack reduced MLP F1 from 0.936 to 0.507,
  a drop of 45.8%. Although MLP has gradients available (enabling white-box
  attacks), the model itself lacks resistance to small perturbations.
```
