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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Create results directory (under baseline/)
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("=" * 60)
print("Baseline: Random Forest + MLP Comparison")
print("=" * 60)

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
print(f"Label distribution: Benign={sum(df_merged['label_binary'] == 0):,}, Malicious={sum(df_merged['label_binary'] == 1):,}")

# ============================================
# 3. Sampling
# ============================================
print("\n[3] Sampling...")

SAMPLE_SIZE = 100000
benign = df_merged[df_merged['label_binary'] == 0].sample(
    n=min(SAMPLE_SIZE, len(df_merged[df_merged['label_binary'] == 0])), random_state=42)
malicious = df_merged[df_merged['label_binary'] == 1].sample(
    n=min(SAMPLE_SIZE, len(df_merged[df_merged['label_binary'] == 1])), random_state=42)
df_sample = pd.concat([benign, malicious], ignore_index=True)
print(f"Sampled data shape: {df_sample.shape}")

# ============================================
# 4. Prepare data
# ============================================
print("\n[4] Prepare training data...")

feature_cols = [c for c in df_sample.columns if c not in ['label_binary', 'label']]
X = df_sample[feature_cols].values
y = df_sample['label_binary'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")


# ============================================
# 5. MLP Model Definition
# ============================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[128, 64]):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPWrapper:
    def __init__(self, input_dim, hidden_layers=[128, 64], lr=0.001, epochs=30, batch_size=512):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = MLP(self.input_dim, self.hidden_layers).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


# ============================================
# 6. Train Random Forest
# ============================================
print("\n[5] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf)
}
print(f"RF: Acc={rf_metrics['accuracy']:.4f}, F1={rf_metrics['f1']:.4f}")

# ============================================
# 7. Train MLP
# ============================================
print("\n[6] Training MLP...")
mlp = MLPWrapper(input_dim=X_train.shape[1], hidden_layers=[128, 64], lr=0.001, epochs=30)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

mlp_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_mlp),
    'f1': f1_score(y_test, y_pred_mlp)
}
print(f"MLP: Acc={mlp_metrics['accuracy']:.4f}, F1={mlp_metrics['f1']:.4f}")

# ============================================
# 8. Save results
# ============================================
print("\n[7] Saving results...")

comparison_df = pd.DataFrame([
    {'model': 'RandomForest', 'accuracy': rf_metrics['accuracy'], 'f1': rf_metrics['f1']},
    {'model': 'MLP', 'accuracy': mlp_metrics['accuracy'], 'f1': mlp_metrics['f1']}
])
comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
print(f"Results saved to {results_dir}/model_comparison.csv")

print("\n" + "=" * 60)
print("Comparison Results:")
print(f"  Random Forest: Acc={rf_metrics['accuracy']:.4f}, F1={rf_metrics['f1']:.4f}")
print(f"  MLP:           Acc={mlp_metrics['accuracy']:.4f}, F1={mlp_metrics['f1']:.4f}")
print("=" * 60)