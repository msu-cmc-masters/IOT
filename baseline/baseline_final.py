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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
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
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Using device: {device}")

print("=" * 60)
print("Attack Optimized: Remove ts Feature + Enhanced PGD Attack")
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
# 3. Sampling and remove ts feature
# ============================================
print("\n[3] Sampling and remove timestamp feature...")

SAMPLE_SIZE = 50000
benign = df_merged[df_merged['label_binary'] == 0].sample(
    n=min(SAMPLE_SIZE, len(df_merged[df_merged['label_binary'] == 0])), random_state=42)
malicious = df_merged[df_merged['label_binary'] == 1].sample(
    n=min(SAMPLE_SIZE, len(df_merged[df_merged['label_binary'] == 1])), random_state=42)
df_sample = pd.concat([benign, malicious], ignore_index=True)

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
# 5. MLP Model Definition
# ============================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32, 16]):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPWrapper:
    def __init__(self, input_dim, hidden_layers=[64, 32, 16], lr=0.001, epochs=30, batch_size=256):
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

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"    Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

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
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
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
mlp = MLPWrapper(input_dim=X_train.shape[1], hidden_layers=[64, 32, 16], lr=0.001, epochs=30)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

mlp_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_mlp),
    'f1': f1_score(y_test, y_pred_mlp)
}
print(f"MLP: Acc={mlp_metrics['accuracy']:.4f}, F1={mlp_metrics['f1']:.4f}")

# ============================================
# 8. PGD Attack
# ============================================
print("\n[7] Enhanced PGD Attack...")


def pgd_attack(model, X, epsilon, alpha, num_iter, target_label=0):
    """PGD attack: make model predict target_label (0=Benign)"""
    model.model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    X_adv = X_tensor.clone().detach()

    y_tensor = torch.LongTensor([target_label] * len(X)).to(device)

    for i in range(num_iter):
        X_adv.requires_grad = True

        outputs = model.model(X_adv)
        loss = nn.CrossEntropyLoss()(outputs, y_tensor)

        model.model.zero_grad()
        loss.backward()

        grad_sign = X_adv.grad.data.sign()
        X_adv = X_adv + alpha * grad_sign

        eta = torch.clamp(X_adv - X_tensor, -epsilon, epsilon)
        X_adv = torch.clamp(X_tensor + eta, -5, 5)

        X_adv = X_adv.detach()

    return X_adv.detach().cpu().numpy()


malicious_mask = y_test == 1
X_malicious = X_test[malicious_mask]
print(f"  Malicious samples: {X_malicious.shape[0]}")

epsilons = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
alpha = 0.05
num_iter = 100

attack_results = []

print("\n  Scanning different epsilon values (PGD, iter=100):")
for epsilon in epsilons:
    print(f"    Testing ε={epsilon:.1f}:")
    X_adv = pgd_attack(mlp, X_malicious, epsilon, alpha, num_iter, target_label=0)
    y_pred_adv = mlp.predict(X_adv)
    success_rate = np.mean(y_pred_adv == 0)
    attack_results.append({'epsilon': epsilon, 'success_rate': success_rate})
    print(f"    Final ε={epsilon:.1f}: Attack success rate={success_rate:.2%}")

best_epsilon = max(attack_results, key=lambda x: x['success_rate'])['epsilon']

print(f"\n  Selected best ε={best_epsilon}")

X_adv_best = pgd_attack(mlp, X_malicious, best_epsilon, alpha, num_iter, target_label=0)
y_pred_adv_best = mlp.predict(X_adv_best)

X_test_adv = X_test.copy()
X_test_adv[malicious_mask] = X_adv_best
y_pred_after = mlp.predict(X_test_adv)

after_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_after),
    'f1': f1_score(y_test, y_pred_after)
}

print(f"\n  MLP Metrics After Attack:")
print(f"    Accuracy: {after_metrics['accuracy']:.4f}")
print(f"    F1-Score: {after_metrics['f1']:.4f}")

# ============================================
# 9. Save results and plots
# ============================================
print("\n[8] Saving results...")

# Save attack results plot
plt.figure(figsize=(10, 6))
eps = [r['epsilon'] for r in attack_results]
rates = [r['success_rate'] * 100 for r in attack_results]
plt.plot(eps, rates, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=50, color='r', linestyle='--', label='50% Success Threshold')
plt.xlabel('Epsilon (ε)')
plt.ylabel('Attack Success Rate (%)')
plt.title(f'PGD Attack Success Rate vs Epsilon (iter={num_iter})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(plots_dir, 'pgd_attack_results_optimized.png'), dpi=150, bbox_inches='tight')
plt.close()

# Save performance comparison plot
metrics_names = ['Accuracy', 'F1-Score']
before_values = [mlp_metrics['accuracy'] * 100, mlp_metrics['f1'] * 100]
after_values = [after_metrics['accuracy'] * 100, after_metrics['f1'] * 100]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width / 2, before_values, width, label='Before Attack', color='green')
ax.bar(x + width / 2, after_values, width, label='After Attack', color='red')
ax.set_ylabel('Score (%)')
ax.set_title('MLP Performance Before and After PGD Attack')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'attack_performance_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# Save CSV results
comparison_df = pd.DataFrame([
    {'model': 'RandomForest', 'accuracy': rf_metrics['accuracy'], 'f1': rf_metrics['f1']},
    {'model': 'MLP_Before_Attack', 'accuracy': mlp_metrics['accuracy'], 'f1': mlp_metrics['f1']},
    {'model': 'MLP_After_PGD', 'accuracy': after_metrics['accuracy'], 'f1': after_metrics['f1']}
])
comparison_df.to_csv(os.path.join(results_dir, 'model_comparison_attack.csv'), index=False)

attack_df = pd.DataFrame(attack_results)
attack_df.to_csv(os.path.join(results_dir, 'pgd_results_optimized.csv'), index=False)

print(f"Results saved to {results_dir}/")
print(f"Plots saved to {plots_dir}/")

print("\n" + "=" * 60)
print("Attack Optimized Baseline Complete!")
print("=" * 60)

print("\nFinal Comparison:")
print(f"  Random Forest:         Acc={rf_metrics['accuracy']:.4f}, F1={rf_metrics['f1']:.4f}")
print(f"  MLP (Before Attack):   Acc={mlp_metrics['accuracy']:.4f}, F1={mlp_metrics['f1']:.4f}")
print(f"  MLP (After PGD):       Acc={after_metrics['accuracy']:.4f}, F1={after_metrics['f1']:.4f}")