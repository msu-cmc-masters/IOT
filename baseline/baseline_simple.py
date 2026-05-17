import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 获取当前文件所在目录（baseline/）
current_dir = os.path.dirname(os.path.abspath(__file__))

print("="*50)
print("Step 1: 加载数据")
print("="*50)

# 读取 data 目录下的文件（上级目录的data）
file_path = os.path.join(current_dir, '..', 'data', 'CTU-IoT-Malware-Capture-3-1conn.log.labeled.csv')
df = pd.read_csv(file_path, sep='|')
print(f"数据形状: {df.shape}")

print("\n" + "="*50)
print("Step 2: 数据预处理")
print("="*50)

# 删除无关列
drop_cols = ['uid', 'id.orig_h', 'id.resp_h', 'tunnel_parents', 'detailed-label']
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols, errors='ignore')
print(f"删除无关列后: {df.shape}")

# 转换数值列
numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
numeric_cols = [c for c in numeric_cols if c in df.columns]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].replace('-', np.nan), errors='coerce')
    df[col] = df[col].fillna(0)
print(f"数值列处理完成")

# 编码分类列
categorical_cols = ['proto', 'service', 'conn_state', 'local_orig', 'local_resp', 'history']
categorical_cols = [c for c in categorical_cols if c in df.columns]
for col in categorical_cols:
    df[col] = df[col].fillna('unknown')
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
print(f"分类列编码完成")

# 标签二值化
df['label_binary'] = (df['label'] != 'Benign').astype(int)
print(f"标签分布: 正常={sum(df['label_binary']==0)}, 恶意={sum(df['label_binary']==1)}")

print("\n" + "="*50)
print("Step 3: 采样")
print("="*50)

# 采样（各取20000条）
benign = df[df['label_binary'] == 0].sample(n=min(20000, len(df[df['label_binary']==0])), random_state=42)
malicious = df[df['label_binary'] == 1].sample(n=min(20000, len(df[df['label_binary']==1])), random_state=42)
df_sample = pd.concat([benign, malicious], ignore_index=True)
print(f"采样后数据: {df_sample.shape}")

print("\n" + "="*50)
print("Step 4: 训练模型")
print("="*50)

# 准备特征和标签
feature_cols = [c for c in df_sample.columns if c not in ['label_binary', 'label']]
X = df_sample[feature_cols].values
y = df_sample['label_binary'].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 训练 Random Forest
print("\n训练 Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_test)

print("\n" + "="*50)
print("Step 5: 结果")
print("="*50)
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"F1分数 (F1-Score): {f1_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))

print("\n" + "="*50)
print("Baseline 运行完成！")
print("="*50)