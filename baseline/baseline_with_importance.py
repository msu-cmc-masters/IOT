import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Create results and plots directories (under baseline/)
results_dir = os.path.join(current_dir, 'results')
plots_dir = os.path.join(current_dir, 'plots')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("="*60)
print("Baseline: Random Forest + MLP + Permutation Importance")
print("="*60)

# ============================================
# 1. Read all CSV files
# ============================================
print("\n[1] Reading all CSV files...")

data_path = os.path.join(project_root, 'data')
all_files = glob.glob(os.path.join(data_path, '*.csv'))
print(f"Found {len(all_files)} CSV files")

dfs = []
for file in all_files:
    print(f"  Reading: {os.path.basename(file)}")
    df = pd.read_csv(file, sep='|')
    dfs.append(df)

df_merged = pd.concat(dfs, ignore_index=True)
print(f"Merged data shape: {df_merged.shape}")

# ============================================
# 2. Data preprocessing
# ============================================
print("\n[2] Data preprocessing...")

drop_cols = ['uid', 'id.orig_h', 'id.resp_h', 'tunnel_parents', 'detailed-label']
drop_cols = [c for c in drop_cols if c in df_merged.columns]
df_merged = df_merged.drop(columns=drop_cols, errors='ignore')

numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
numeric_cols = [c for c in numeric_cols if c in df_merged.columns]

for col in numeric_cols:
    df_merged[col] = pd.to_numeric(df_merged[col].replace('-', np.nan), errors='coerce')
    df_merged[col] = df_merged[col].fillna(0)

categorical_cols = ['proto', 'service', 'conn_state', 'local_orig', 'local_resp', 'history']
categorical_cols = [c for c in categorical_cols if c in df_merged.columns]

for col in categorical_cols:
    df_merged[col] = df_merged[col].fillna('unknown')
    df_merged[col] = LabelEncoder().fit_transform(df_merged[col].astype(str))

df_merged['label_binary'] = (df_merged['label'] != 'Benign').astype(int)
print(f"Label distribution: Benign={sum(df_merged['label_binary']==0):,}, Malicious={sum(df_merged['label_binary']==1):,}")

# ============================================
# 3. Sampling
# ============================================
print("\n[3] Sampling...")

SAMPLE_SIZE = 100000
benign = df_merged[df_merged['label_binary'] == 0].sample(n=min(SAMPLE_SIZE, len(df_merged[df_merged['label_binary']==0])), random_state=42)
malicious = df_merged[df_merged['label_binary'] == 1].sample(n=min(SAMPLE_SIZE, len(df_merged[df_merged['label_binary']==1])), random_state=42)
df_sample = pd.concat([benign, malicious], ignore_index=True)

# ========== Critical fix: Remove ts feature ==========
if 'ts' in df_sample.columns:
    df_sample = df_sample.drop(columns=['ts'])
    print("  Removed ts (timestamp) feature")

print(f"Sampled data shape: {df_sample.shape}")

# ============================================
# 4. Prepare data
# ============================================
print("\n[4] Prepare training data...")

feature_cols = [c for c in df_sample.columns if c not in ['label_binary', 'label']]
feature_names = feature_cols
X = df_sample[feature_cols].values
y = df_sample['label_binary'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ============================================
# 5. Train Random Forest
# ============================================
print("\n[5] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"RF: Acc={accuracy_score(y_test, y_pred_rf):.4f}, F1={f1_score(y_test, y_pred_rf):.4f}")

# ============================================
# 6. Permutation Importance
# ============================================
print("\n[6] Computing Permutation Importance...")

result = permutation_importance(rf, X_test, y_test, n_repeats=10, scoring='accuracy', random_state=42)
sorted_idx = np.argsort(result.importances_mean)[::-1]

print("\nTop-10 Important Features:")
for i, idx in enumerate(sorted_idx[:min(10, len(feature_names))]):
    print(f"  {i+1}. {feature_names[idx]}: {result.importances_mean[idx]:.4f}")

# Save feature importance plot
top_n = min(20, len(feature_names))
plt.figure(figsize=(10, max(6, top_n * 0.3)))
plt.barh(range(top_n), result.importances_mean[sorted_idx[:top_n]], capsize=3)
plt.yticks(range(top_n), [feature_names[idx] for idx in sorted_idx[:top_n]])
plt.xlabel('Permutation Importance')
plt.title('Random Forest - Feature Importance (ts removed)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'permutation_importance_rf.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 7. Save results
# ============================================
print("\n[7] Saving results...")

importance_df = pd.DataFrame({
    'feature': [feature_names[idx] for idx in sorted_idx],
    'importance': result.importances_mean[sorted_idx]
})
importance_df.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
print(f"Results saved to {results_dir}/")

print("\n" + "="*60)
print("Complete!")
print("="*60)