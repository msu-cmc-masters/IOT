import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Get current file directory (baseline/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Create results directory (under baseline/)
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

print("="*60)
print("Full Baseline: Read All CSV Files")
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

# Drop irrelevant columns
drop_cols = ['uid', 'id.orig_h', 'id.resp_h', 'tunnel_parents', 'detailed-label']
drop_cols = [c for c in drop_cols if c in df_merged.columns]
df_merged = df_merged.drop(columns=drop_cols, errors='ignore')
print(f"After dropping columns: {df_merged.shape}")

# Convert numeric columns
numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
numeric_cols = [c for c in numeric_cols if c in df_merged.columns]

for col in numeric_cols:
    df_merged[col] = pd.to_numeric(df_merged[col].replace('-', np.nan), errors='coerce')
    df_merged[col] = df_merged[col].fillna(0)
print(f"Numeric columns processed")

# Encode categorical columns
categorical_cols = ['proto', 'service', 'conn_state', 'local_orig', 'local_resp', 'history']
categorical_cols = [c for c in categorical_cols if c in df_merged.columns]

for col in categorical_cols:
    df_merged[col] = df_merged[col].fillna('unknown')
    df_merged[col] = LabelEncoder().fit_transform(df_merged[col].astype(str))
print(f"Categorical columns encoded")

# Binary label
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
print(f"Sampled data shape: {df_sample.shape}")

# ============================================
# 4. Prepare data
# ============================================
print("\n[4] Prepare training data...")

feature_cols = [c for c in df_sample.columns if c not in ['label_binary', 'label']]
X = df_sample[feature_cols].values
y = df_sample['label_binary'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ============================================
# 5. Train Random Forest
# ============================================
print("\n[5] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nRandom Forest Results:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")

# ============================================
# 6. Save results
# ============================================
print("\n[6] Saving results...")

metrics_df = pd.DataFrame([{
    'model': 'RandomForest',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'train_size': len(X_train),
    'test_size': len(X_test)
}])
metrics_df.to_csv(os.path.join(results_dir, 'baseline_metrics.csv'), index=False)
print(f"Metrics saved to {results_dir}/baseline_metrics.csv")

print("\n" + "="*60)
print("Full Baseline Complete!")
print("="*60)