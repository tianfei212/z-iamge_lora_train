import os, json, torch, time
import torch.multiprocessing as mp
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig

# ================= 配置区域 =================
# 1. 基础模型路径
MODEL_ROOT = "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo"
DEFAULT_MODEL_PATHS = json.dumps([
    [f"{MODEL_ROOT}/transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
     f"{MODEL_ROOT}/transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
     f"{MODEL_ROOT}/transformer/diffusion_pytorch_model-00003-of-00003.safetensors"],
    [f"{MODEL_ROOT}/text_encoder/model-00001-of-00003.safetensors",
     f"{MODEL_ROOT}/text_encoder/model-00002-of-00003.safetensors",
     f"{MODEL_ROOT}/text_encoder/model-00003-of-00003.safetensors"],
    f"{MODEL_ROOT}/vae/diffusion_pytorch_model.safetensors",
])
TOKENIZER_PATH = f"{MODEL_ROOT}/tokenizer"

# 2. LoRA 路径
CHAR_LORA_PATH = "/home/ubuntu/codes/wan_lora/output/lyf_zimage_turbo_lora_v1/step-1400.safetensors"
TRIGGER_WORD = "ohwx woman"
DISTILL_LORA_PATH = "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo-DistillPatch/model.safetensors"

# 3. 输出路径
OUTPUT_DIR = "/home/ubuntu/codes/wan_lora/output/zimage_lora_compare_step1400"

# 4. 参数配置
NUM_STEPS = 8          # Turbo 模式建议 4-8 步
CFG_SCALE = 1.5        # Turbo 模式建议降低 CFG
HEIGHT = 1024
WIDTH = 1024

# 5. 测试 Prompt 列表
TEST_PROMPTS = [
    "A simple portrait of a woman, white t-shirt, natural lighting, looking at viewer",
    "A professional business woman wearing a suit, standing in a modern office, confidence",
    "A woman wearing traditional Chinese Hanfu, ancient garden background, soft lighting, ethereal",
    "A glamorous woman in an elegant evening gown, red carpet event, flashing cameras, jewelry",
    "A cyberpunk female warrior, neon lights, rainy city street, futuristic armor, mechanical details",
    "Extreme close-up side profile of a woman, detailed skin texture, cinematic lighting, dark background",
    "A fantasy elf woman with long ears, magic forest, glowing particles, dreamlike atmosphere",
    "A woman sitting in a coffee shop, reading a book, cozy autumn vibe, wearing a sweater",
    "A fitness woman jogging in the park, sportswear, sunny day, dynamic pose",
    "Oil painting of a noble woman, vintage style, thick brushstrokes, by John Singer Sargent"
]
# ===========================================

def parse_model_configs(model_paths_json: str):
    model_paths = json.loads(model_paths_json)
    return [ModelConfig(path=path) for path in model_paths]

def gpu_worker(rank, world_size):
    device = f"cuda:{rank}"
    
    # === 角色分配 ===
    # rank 0: Baseline (Group A)
    # rank 1: Experiment (Group B)
    if rank == 0:
        role = "A_Group (Baseline)"
        use_char_lora = False
        save_subfolder = "no_trigger"
        print(f"[GPU 0] Role Assigned: {role} - Running purely on DistillPatch")
    else:
        role = "B_Group (Liuyifei)"
        use_char_lora = True
        save_subfolder = "with_trigger"
        print(f"[GPU 1] Role Assigned: {role} - Running with DistillPatch + Character LoRA")

    # 1. 初始化底模
    print(f"[GPU {rank}] Initializing Base Model...")
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=parse_model_configs(DEFAULT_MODEL_PATHS),
        tokenizer_config=ModelConfig(TOKENIZER_PATH),
    )

    # 2. 所有人必须加载加速 LoRA (DistillPatch)
    print(f"[GPU {rank}] Loading DistillPatch (Acceleration)...")
    try:
        pipe.load_lora(pipe.dit, DISTILL_LORA_PATH, alpha=1.0)
    except AttributeError:
        pipe.load_lora(pipe.transformer, DISTILL_LORA_PATH, alpha=1.0)

    # 3. 只有 B 组 (GPU 1) 加载角色 LoRA
    if use_char_lora:
        print(f"[GPU {rank}] Loading Character LoRA ({CHAR_LORA_PATH})...")
        try:
            pipe.load_lora(pipe.dit, CHAR_LORA_PATH, alpha=0.8)
        except AttributeError:
            pipe.load_lora(pipe.transformer, CHAR_LORA_PATH, alpha=0.8)

    print(f"[GPU {rank}] Ready! Processing all {len(TEST_PROMPTS)} prompts...")

    # 4. 开始生成循环
    # 注意：这里不再对 prompts 进行切片，每张卡都要跑完整的 10 个 prompt
    for i, base_prompt in enumerate(TEST_PROMPTS):
        
        # 构建 Prompt 和 Seed
        seed = 1000 + i  # 【关键】两张卡使用完全相同的 Seed
        
        if use_char_lora:
            # B组：加触发词
            prompt = f"{TRIGGER_WORD}, {base_prompt}"
        else:
            # A组：原Prompt
            prompt = base_prompt

        start_t = time.time()
        
        # 生成
        image = pipe(
            prompt=prompt,
            negative_prompt="blur, low quality, distortion, ugly",
            cfg_scale=CFG_SCALE,
            num_inference_steps=NUM_STEPS,
            height=HEIGHT,
            width=WIDTH,
            seed=seed
        )
        
        duration = time.time() - start_t
        
        # 保存
        # 文件名保持一致，方便最后直接肉眼对比
        filename = f"{i+1:02d}_seed{seed}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_subfolder, filename)
        image.save(save_path)
        
        print(f"[GPU {rank}] {role} | Img {i+1}/{len(TEST_PROMPTS)} | {duration:.2f}s | Saved: {filename}")

    print(f"[GPU {rank}] ✅ Mission Complete.")

def main():
    # 准备目录
    os.makedirs(os.path.join(OUTPUT_DIR, "no_trigger"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "with_trigger"), exist_ok=True)
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Error: Need at least 2 GPUs for this split test.")
        return

    print(f"🚀 Launching Split A/B Test on {world_size} GPUs...")
    print("   - GPU 0: Group A (Base + Distill)")
    print("   - GPU 1: Group B (Base + Distill + Liuyifei)")
    
    mp.spawn(gpu_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()