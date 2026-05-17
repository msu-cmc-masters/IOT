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

# 获取当前文件所在目录（baseline/）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 创建结果目录（在 baseline 下）
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

print("="*60)
print("完整 Baseline: 读取所有 CSV 文件")
print("="*60)

# ============================================
# 1. 读取所有 CSV 文件
# ============================================
print("\n[1] 读取所有 CSV 文件...")

data_path = os.path.join(project_root, 'data')
all_files = glob.glob(os.path.join(data_path, '*.csv'))
print(f"找到 {len(all_files)} 个 CSV 文件")

dfs = []
for file in all_files:
    print(f"  读取: {os.path.basename(file)}")
    df = pd.read_csv(file, sep='|')
    dfs.append(df)

df_merged = pd.concat(dfs, ignore_index=True)
print(f"合并后数据形状: {df_merged.shape}")

# ============================================
# 2. 数据预处理
# ============================================
print("\n[2] 数据预处理...")

# 删除无关列
drop_cols = ['uid', 'id.orig_h', 'id.resp_h', 'tunnel_parents', 'detailed-label']
drop_cols = [c for c in drop_cols if c in df_merged.columns]
df_merged = df_merged.drop(columns=drop_cols, errors='ignore')
print(f"删除无关列后: {df_merged.shape}")

# 转换数值列
numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
numeric_cols = [c for c in numeric_cols if c in df_merged.columns]

for col in numeric_cols:
    df_merged[col] = pd.to_numeric(df_merged[col].replace('-', np.nan), errors='coerce')
    df_merged[col] = df_merged[col].fillna(0)
print(f"数值列处理完成")

# 编码分类列
categorical_cols = ['proto', 'service', 'conn_state', 'local_orig', 'local_resp', 'history']
categorical_cols = [c for c in categorical_cols if c in df_merged.columns]

for col in categorical_cols:
    df_merged[col] = df_merged[col].fillna('unknown')
    df_merged[col] = LabelEncoder().fit_transform(df_merged[col].astype(str))
print(f"分类列编码完成")

# 标签二值化
df_merged['label_binary'] = (df_merged['label'] != 'Benign').astype(int)
print(f"标签分布: 正常={sum(df_merged['label_binary']==0):,}, 恶意={sum(df_merged['label_binary']==1):,}")

# ============================================
# 3. 采样
# ============================================
print("\n[3] 采样...")

SAMPLE_SIZE = 100000
benign = df_merged[df_merged['label_binary'] == 0].sample(n=min(SAMPLE_SIZE, len(df_merged[df_merged['label_binary']==0])), random_state=42)
malicious = df_merged[df_merged['label_binary'] == 1].sample(n=min(SAMPLE_SIZE, len(df_merged[df_merged['label_binary']==1])), random_state=42)
df_sample = pd.concat([benign, malicious], ignore_index=True)
print(f"采样后: {df_sample.shape}")

# ============================================
# 4. 准备数据
# ============================================
print("\n[4] 准备训练数据...")

feature_cols = [c for c in df_sample.columns if c not in ['label_binary', 'label']]
X = df_sample[feature_cols].values
y = df_sample['label_binary'].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ============================================
# 5. 训练 Random Forest
# ============================================
print("\n[5] 训练 Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nRandom Forest 结果:")
print(f"  准确率: {accuracy:.4f}")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"  F1分数: {f1:.4f}")

# ============================================
# 6. 保存结果
# ============================================
print("\n[6] 保存结果...")

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
print(f"指标已保存到 {results_dir}/baseline_metrics.csv")

print("\n" + "="*60)
print("完整 Baseline 运行完成！")
print("="*60)