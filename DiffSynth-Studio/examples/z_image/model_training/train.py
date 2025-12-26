import torch, os, argparse, accelerate, copy
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ZImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        image_key="image",
        prompt_key="prompt",
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        self.pipe = ZImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        if task == "trajectory_imitation":
            # This is an experimental feature.
            # We may remove it in the future.
            self.loss_fn = TrajectoryImitationLoss()
            self.task_to_loss["trajectory_imitation"] = self.loss_fn
            self.pipe_teacher = copy.deepcopy(self.pipe)
            self.pipe_teacher.requires_grad_(False)
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data[self.prompt_key]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data[self.image_key],
            "height": data[self.image_key].size[1],
            "width": data[self.image_key].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        if self.task == "trajectory_imitation":
            inputs_shared["cfg_scale"] = 2
            inputs_shared["teacher"] = self.pipe_teacher
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss
    
    def sample_and_save(
        self,
        dataset,
        output_path: str,
        step: int,
        sample_num_images: int = 10,
        sample_seed: int = 0,
        sample_inference_steps: int = 8,
        sample_denoising_strength: float = 1.0,
        sample_cfg_scale: float = 1.0,
        sample_negative_prompt: str = "",
        sample_subdir: str = "samples",
    ):
        if step is None:
            return
        folder = os.path.join(output_path, sample_subdir, f"step-{step}")
        os.makedirs(folder, exist_ok=True)
        total = min(int(sample_num_images), len(dataset)) if hasattr(dataset, "__len__") else int(sample_num_images)
        for i in range(total):
            data = dataset[i]
            prompt = str(data[self.prompt_key])
            input_image = data[self.image_key]
            height = input_image.size[1]
            width = input_image.size[0]
            image = self.pipe(
                prompt=prompt,
                negative_prompt=sample_negative_prompt,
                cfg_scale=sample_cfg_scale,
                input_image=input_image,
                denoising_strength=sample_denoising_strength,
                height=height,
                width=width,
                seed=sample_seed + i,
                rand_device=self.pipe.device,
                num_inference_steps=sample_inference_steps,
            )
            image.save(os.path.join(folder, f"{i:03d}.png"))
        self.pipe.scheduler.set_timesteps(1000, training=True)


def z_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--image_key", type=str, default="image", help="Metadata key for the input image path.")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Metadata key for the prompt.")
    parser.add_argument("--sample_num_images", type=int, default=10, help="Number of sample images per checkpoint. Set 0 to disable.")
    parser.add_argument("--sample_seed", type=int, default=0, help="Seed base for sampling. Actual seed is seed + index.")
    parser.add_argument("--sample_inference_steps", type=int, default=8, help="Number of inference steps for sampling.")
    parser.add_argument("--sample_denoising_strength", type=float, default=1.0, help="Denoising strength for sampling.")
    parser.add_argument("--sample_cfg_scale", type=float, default=1.0, help="CFG scale for sampling.")
    parser.add_argument("--sample_negative_prompt", type=str, default="", help="Negative prompt for sampling.")
    parser.add_argument("--sample_subdir", type=str, default="samples", help="Subdir under output_path for saving samples.")
    parser.set_defaults(data_file_keys="image")
    return parser


if __name__ == "__main__":
    parser = z_image_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    model = ZImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        image_key=args.image_key,
        prompt_key=args.prompt_key,
        device=accelerator.device,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "trajectory_imitation": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
