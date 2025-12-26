from modelscope import snapshot_download
import os

# 目标路径
target_dir = "/home/ubuntu/codes/wan_lora/models/Wan2.2-TI2V-5B"
os.makedirs(target_dir, exist_ok=True)

print("🚀 开始下载 Wan2.2-TI2V-5B (720P 高速版)...")
# 从魔搭社区下载
snapshot_download('Wan-AI/Wan2.2-TI2V-5B', local_dir=target_dir)
print(f"✅ 下载完成！模型存放于: {target_dir}")