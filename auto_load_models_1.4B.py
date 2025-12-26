from modelscope import snapshot_download
import os

# 目标路径修改为 Z-Image 专用目录
target_dir = "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo"
os.makedirs(target_dir, exist_ok=True)

print("🚀 开始下载 Tongyi-MAI/Z-Image-Turbo (1.4B 图像极致加速版)...")

# 使用 Z-Image-Turbo 正式 ID
snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir=target_dir)

print(f"✅ 下载完成！模型存放于: {target_dir}")