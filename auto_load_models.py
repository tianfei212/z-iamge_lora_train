from modelscope import snapshot_download
import os

# 目标路径
target_dir = "/home/ubuntu/codes/wan_lora/models/Wan2.2-T2V-A14B"
os.makedirs(target_dir, exist_ok=True)

print("🚀 开始下载 Wan2.2-T2V-A14B 权重...")
# 下载并保存到你的 models 目录
snapshot_download('Wan-AI/Wan2.2-T2V-A14B', local_dir=target_dir)
print(f"✅ 下载完成！路径: {target_dir}")