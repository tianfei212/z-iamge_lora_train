import os, argparse, json, torch
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig


def parse_model_configs(model_paths_json: str):
    model_paths = json.loads(model_paths_json)
    return [ModelConfig(path=path) for path in model_paths]


def get_default_angles():
    return [
        "front view",
        "three-quarter view from the left",
        "three-quarter view from the right",
        "profile view",
        "back view",
        "high angle shot",
        "low angle shot",
        "top-down view",
        "close-up portrait",
        "wide shot",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, required=True, help="JSON list, same format as training --model_paths")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--rand_device", type=str, default=None)

    parser.add_argument("--base_prompt", type=str, required=True)
    parser.add_argument("--lora_trigger", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rand_device = args.rand_device or device
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.torch_dtype]

    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=parse_model_configs(args.model_paths),
        tokenizer_config=ModelConfig(args.tokenizer_path),
    )
    pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)

    out_no = os.path.join(args.output_dir, "no_trigger")
    out_yes = os.path.join(args.output_dir, "with_trigger")
    os.makedirs(out_no, exist_ok=True)
    os.makedirs(out_yes, exist_ok=True)

    angles = get_default_angles()
    for i, angle in enumerate(angles):
        prompt = f"{args.base_prompt}, {angle}"
        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            cfg_scale=args.cfg_scale,
            height=args.height,
            width=args.width,
            seed=args.seed + i,
            rand_device=rand_device,
            num_inference_steps=args.steps,
        )
        image.save(os.path.join(out_no, f"{i:03d}.png"))

    for i, angle in enumerate(angles):
        prompt = f"{args.base_prompt}, {args.lora_trigger}, {angle}"
        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            cfg_scale=args.cfg_scale,
            height=args.height,
            width=args.width,
            seed=args.seed + 1000 + i,
            rand_device=rand_device,
            num_inference_steps=args.steps,
        )
        image.save(os.path.join(out_yes, f"{i:03d}.png"))


if __name__ == "__main__":
    main()

