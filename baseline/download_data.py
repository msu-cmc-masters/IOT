import kagglehub
import os
import shutil

# 获取项目根目录（baseline的上级目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 数据目录（在项目根目录下）
DATA_DIR = os.path.join(project_root, '../data')
os.makedirs(DATA_DIR, exist_ok=True)

print("正在下载数据集...")
path = kagglehub.dataset_download("agungpambudi/network-malware-detection-connection-analysis")
print(f"下载完成: {path}")

# 拷贝 CSV 文件
for file in os.listdir(path):
    if file.endswith('.csv'):
        src = os.path.join(path, file)
        dst = os.path.join(DATA_DIR, file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"已拷贝: {file}")

print(f"数据已保存到: {DATA_DIR}")