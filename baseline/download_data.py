# 下载脚本（已在之前的消息中提供）
import kagglehub
import os
import shutil

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

path = kagglehub.dataset_download("agungpambudi/network-malware-detection-connection-analysis")

for file in os.listdir(path):
    if file.endswith('.csv'):
        src = os.path.join(path, file)
        dst = os.path.join(DATA_DIR, file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"已拷贝 {file}")