import os
# 关键：跳过 Torch 2.6 的强制版本检查
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERIFY_SCHEDULED_DOWNLOAD"] = "0"
import torch
torch.serialization.add_safe_globals([set]) # 允许加载基本集合
from PIL import Image
from transformers import pipeline

# 1. 路径与参数配置
INPUT_DIR = "/home/ubuntu/codes/wan_lora/data/images"
OUTPUT_DIR = "/home/ubuntu/codes/wan_lora/data/captions"
TRIGGER_WORD = "ohwx woman"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 定义你想要存放模型的路径
MODEL_CACHE_DIR = "/home/ubuntu/codes/wan_lora/models/tagger_model"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
def process_images_and_tag(input_dir, output_dir, trigger_word):
    # 初始化 AI 打标器
    print(f"正在加载 AI 打标模型至: {MODEL_CACHE_DIR}")
    device = 0 if torch.cuda.is_available() else -1
    # 使用 image-to-text 流水线
    captioner = pipeline(
    "image-to-text", 
    model="Salesforce/blip-image-captioning-base", 
    device=device,
    use_safetensors=True,  # 显式指定使用 safetensors
    model_kwargs={"cache_dir": MODEL_CACHE_DIR}
)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"找到 {len(files)} 张图片，开始处理...")

    for filename in files:
        # --- 1. 图像处理部分：保证 64 倍数与比例 ---
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")
        
        w, h = img.size
        # 强制缩放到长边 1024，并确保是 64 的倍数
        scale = 1024 / max(w, h)
        new_w = int((w * scale) // 64 * 64)
        new_h = int((h * scale) // 64 * 64)
        
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 覆写原图以节省空间，或你可以另存到新目录
        img_resized.save(img_path, quality=95)

        # --- 2. 自动化打标部分：根据画面生成内容 ---
        try:
            # AI 分析处理后的图片内容
            result = captioner(img_path)
            ai_description = result[0]['generated_text']
            
            # 专业的打标格式：[触发词] + [AI描述] + [质量词]
            final_caption = f"A photo of {trigger_word}, {ai_description}, high quality, cinematic lighting."
        except Exception as e:
            print(f"打标失败 {filename}: {e}")
            final_caption = f"A photo of {trigger_word}, high quality, cinematic lighting."

        # 保存为 .txt 文件
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(final_caption)
            
        print(f"完成: {filename} -> {final_caption}")

    print(f"\n✅ 处理全部完成！")
    print(f"图片已更新并对齐，标签文件位于: {output_dir}")

if __name__ == "__main__":
    process_images_and_tag(INPUT_DIR, OUTPUT_DIR, TRIGGER_WORD)