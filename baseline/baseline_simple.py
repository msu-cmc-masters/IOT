import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Get current file directory (baseline/)
current_dir = os.path.dirname(os.path.abspath(__file__))

print("="*50)
print("Step 1: Load Data")
print("="*50)

# Read file from data directory (parent directory's data)
file_path = os.path.join(current_dir, '..', 'data', 'CTU-IoT-Malware-Capture-3-1conn.log.labeled.csv')
df = pd.read_csv(file_path, sep='|')
print(f"Data shape: {df.shape}")

print("\n" + "="*50)
print("Step 2: Data Preprocessing")
print("="*50)

# Drop irrelevant columns
drop_cols = ['uid', 'id.orig_h', 'id.resp_h', 'tunnel_parents', 'detailed-label']
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols, errors='ignore')
print(f"After dropping columns: {df.shape}")

# Convert numeric columns
numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
numeric_cols = [c for c in numeric_cols if c in df.columns]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].replace('-', np.nan), errors='coerce')
    df[col] = df[col].fillna(0)
print(f"Numeric columns processed")

# Encode categorical columns
categorical_cols = ['proto', 'service', 'conn_state', 'local_orig', 'local_resp', 'history']
categorical_cols = [c for c in categorical_cols if c in df.columns]
for col in categorical_cols:
    df[col] = df[col].fillna('unknown')
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
print(f"Categorical columns encoded")

# Binary label
df['label_binary'] = (df['label'] != 'Benign').astype(int)
print(f"Label distribution: Benign={sum(df['label_binary']==0)}, Malicious={sum(df['label_binary']==1)}")

print("\n" + "="*50)
print("Step 3: Sampling")
print("="*50)

# Sample (20,000 per class)
benign = df[df['label_binary'] == 0].sample(n=min(20000, len(df[df['label_binary']==0])), random_state=42)
malicious = df[df['label_binary'] == 1].sample(n=min(20000, len(df[df['label_binary']==1])), random_state=42)
df_sample = pd.concat([benign, malicious], ignore_index=True)
print(f"Sampled data shape: {df_sample.shape}")

print("\n" + "="*50)
print("Step 4: Train Model")
print("="*50)

# Prepare features and labels
feature_cols = [c for c in df_sample.columns if c not in ['label_binary', 'label']]
X = df_sample[feature_cols].values
y = df_sample['label_binary'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

print("\n" + "="*50)
print("Step 5: Results")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))

print("\n" + "="*50)
print("Baseline Complete!")
print("="*50)