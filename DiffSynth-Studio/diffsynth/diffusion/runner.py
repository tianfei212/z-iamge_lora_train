import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                loss_tensor = loss.detach().float()
                if hasattr(accelerator, "gather_for_metrics"):
                    loss_for_metrics = accelerator.gather_for_metrics(loss_tensor).mean().item()
                else:
                    loss_for_metrics = accelerator.gather(loss_tensor).mean().item()
                lr = float(optimizer.param_groups[0].get("lr", learning_rate))
                save_info = model_logger.on_step_end(
                    accelerator,
                    model,
                    save_steps,
                    logs={"epoch": epoch_id, "loss": loss_for_metrics, "lr": lr},
                )
                if save_info.get("saved") and args is not None:
                    sample_num_images = getattr(args, "sample_num_images", 0)
                    if sample_num_images is not None and int(sample_num_images) > 0:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        if hasattr(unwrapped_model, "sample_and_save") and accelerator.is_main_process:
                            unwrapped_model.sample_and_save(
                                dataset=dataset,
                                output_path=model_logger.output_path,
                                step=save_info.get("step"),
                                sample_num_images=int(sample_num_images),
                                sample_seed=getattr(args, "sample_seed", 0),
                                sample_inference_steps=getattr(args, "sample_inference_steps", 8),
                                sample_denoising_strength=getattr(args, "sample_denoising_strength", 1.0),
                                sample_cfg_scale=getattr(args, "sample_cfg_scale", 1.0),
                                sample_negative_prompt=getattr(args, "sample_negative_prompt", ""),
                                sample_subdir=getattr(args, "sample_subdir", "samples"),
                            )
                        accelerator.wait_for_everyone()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    end_info = model_logger.on_training_end(accelerator, model, save_steps)
    if end_info.get("saved") and args is not None:
        sample_num_images = getattr(args, "sample_num_images", 0)
        if sample_num_images is not None and int(sample_num_images) > 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "sample_and_save") and accelerator.is_main_process:
                unwrapped_model.sample_and_save(
                    dataset=dataset,
                    output_path=model_logger.output_path,
                    step=end_info.get("step"),
                    sample_num_images=int(sample_num_images),
                    sample_seed=getattr(args, "sample_seed", 0),
                    sample_inference_steps=getattr(args, "sample_inference_steps", 8),
                    sample_denoising_strength=getattr(args, "sample_denoising_strength", 1.0),
                    sample_cfg_scale=getattr(args, "sample_cfg_scale", 1.0),
                    sample_negative_prompt=getattr(args, "sample_negative_prompt", ""),
                    sample_subdir=getattr(args, "sample_subdir", "samples"),
                )
            accelerator.wait_for_everyone()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
