import os
import shutil

PIC_DIR = "pic"
TRAIN_DIR = "training"
TEST_DIR = "testing"
QUERY_FILE = "query.txt"

with open(QUERY_FILE, "r") as f:
    test_indices = set(int(line.strip()) for line in f if line.strip())

subfolders = sorted([d for d in os.listdir(PIC_DIR) if os.path.isdir(os.path.join(PIC_DIR, d))])

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for i, folder in enumerate(subfolders):      # 逐個資料夾處理
    src_folder = os.path.join(PIC_DIR, folder)

    train_folder = os.path.join(TRAIN_DIR, folder)
    test_folder  = os.path.join(TEST_DIR, folder)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for img_idx in range(1, 201):
        global_index = i * 200 + img_idx

        filename = f"{folder}_{img_idx:03d}.jpg"
        src_path = os.path.join(src_folder, filename)

        if global_index in test_indices:
            dst_path = os.path.join(test_folder, filename)
        else:
            dst_path = os.path.join(train_folder, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"檔案不存在: {src_path}")
