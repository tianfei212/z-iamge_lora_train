import os, torch, json
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, logs=None):
        self.num_steps += 1
        if logs is not None:
            logs = dict(logs)
            logs.setdefault("step", self.num_steps)
            self.log_metrics(accelerator, logs)
        saved = False
        checkpoint_path = None
        if save_steps is not None and self.num_steps % save_steps == 0:
            checkpoint_path = self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            saved = True
        return {"saved": saved, "step": self.num_steps, "checkpoint_path": checkpoint_path}


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            return {"saved": True, "epoch": epoch_id, "checkpoint_path": path}
        return {"saved": False, "epoch": epoch_id, "checkpoint_path": None}


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        saved = False
        checkpoint_path = None
        if save_steps is not None and self.num_steps % save_steps != 0:
            checkpoint_path = self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            saved = True
        return {"saved": saved, "step": self.num_steps, "checkpoint_path": checkpoint_path}


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            return path
        return None


    def log_metrics(self, accelerator: Accelerator, metrics: dict):
        if not accelerator.is_main_process:
            return
        os.makedirs(self.output_path, exist_ok=True)
        jsonl_path = os.path.join(self.output_path, "metrics.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if all(k in metrics for k in ("step", "loss", "lr")):
            csv_path = os.path.join(self.output_path, "metrics.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("step,epoch,loss,lr\n")
                epoch = metrics.get("epoch", "")
                f.write(f"{metrics['step']},{epoch},{metrics['loss']},{metrics['lr']}\n")
