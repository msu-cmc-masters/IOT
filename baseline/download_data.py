import kagglehub
import os
import shutil

# Get project root directory (parent directory of baseline)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Data directory (under project root)
DATA_DIR = os.path.join(project_root, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading dataset...")
path = kagglehub.dataset_download("agungpambudi/network-malware-detection-connection-analysis")
print(f"Download complete: {path}")

# Copy CSV files
for file in os.listdir(path):
    if file.endswith('.csv'):
        src = os.path.join(path, file)
        dst = os.path.join(DATA_DIR, file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"Copied: {file}")

print(f"Data saved to: {DATA_DIR}")