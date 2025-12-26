import os
import pandas as pd

# 你的数据路径
data_dir = "/home/ubuntu/codes/wan_lora/data/lyf_dataset"
# 获取所有图片（排除 csv 自己）
images = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 核心数据结构：DiffSynth 需要 file_name 映射到描述词
data = []
for img in images:
    base = os.path.splitext(img)[0]
    txt_file = os.path.join(data_dir, f"{base}.txt")
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        data.append({"file_name": img, "text": caption})

# 保存为 metadata.csv
df = pd.DataFrame(data)
df.to_csv(os.path.join(data_dir, "metadata.csv"), index=False)
print(f"✅ 成功索引 {len(data)} 张图片到 metadata.csv")