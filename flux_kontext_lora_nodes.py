# ==================================================================================================
#            START OF THE SINGLE PYTHON FILE: flux_kontext_lora_nodes.py (PRE-STITCHED VERSION)
# ==================================================================================================
# This script is specifically modified to train a face-swapping LoRA using a PRE-STITCHED
# control image, as per the user's improved workflow design.
# ==================================================================================================

import torch
import os
import json
import glob
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps
import hashlib
import shutil
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

# --- Required Imports ---
try:
    import folder_paths
    from peft import LoraConfig, get_peft_model_state_dict
    from diffusers import AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig, T5EncoderModel, CLIPTextModel
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    import bitsandbytes as bnb
    
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin
    from diffusers.utils import logging
    print("--> FLUX Kontext LoRA Nodes (Pre-Stitched Method): All required libraries loaded successfully.")
except ImportError as e:
    print(f"FATAL: A required library is not installed: {e}")
    print("Please ensure you have 'diffusers', 'transformers', 'peft', 'bitsandbytes', and 'accelerate' installed in your ComfyUI Python environment.")

logger = logging.get_logger(__name__)

# ==================================================================================================
#              HELPER: Scan for Checkpoint Models to Populate the Dropdown
# ==================================================================================================
try:
    checkpoint_folders = folder_paths.get_folder_paths("checkpoints")
    flux_kontext_model_files = []
    
    for folder in checkpoint_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".safetensors", ".ckpt")):
                    full_path = os.path.join(root, file)
                    flux_kontext_model_files.append(full_path)
    
    flux_kontext_model_files.insert(0, "black-forest-labs/FLUX.1-Kontext-dev")

except Exception as e:
    print(f"Warning: Could not scan for checkpoints. Using default only. Error: {e}")
    flux_kontext_model_files = ["black-forest-labs/FLUX.1-Kontext-dev"]


# ==================================================================================================
#                  COPIED FluxKontextPipeline (required for saving)
# ==================================================================================================
try:
    from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
except ImportError:
    from dataclasses import dataclass
    from diffusers.utils import BaseOutput
    @dataclass
    class FluxPipelineOutput(BaseOutput):
        images: Union[torch.Tensor, np.ndarray, List[Image.Image]]

class FluxKontextPipeline(DiffusionPipeline, FluxLoraLoaderMixin, FromSingleFileMixin):
    def __init__(self, transformer, vae, text_encoder, tokenizer, scheduler, text_encoder_2, tokenizer_2):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2)


# ==================================================================================================
#                                     ALL NODE CLASSES
# ==================================================================================================

class LoadFluxKontextForLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "model_path": (flux_kontext_model_files, ), "lora_rank": ("INT", {"default": 32, "min": 1}), "lora_alpha": ("FLOAT", {"default": 32.0, "min": 0.1, "step": 0.1}), "quantize_to_8bit": ("BOOLEAN", {"default": True}), "lora_target_modules": ("STRING", {"multiline": True, "default": "attn.to_q\nattn.to_k\nattn.to_v\nattn.to_out.0\nff.net.0.proj\nff.net.2"}), } }
    RETURN_TYPES = ("MODEL",); RETURN_NAMES = ("flux_transformer",); FUNCTION = "load_model"; CATEGORY = "FLUX Kontext LoRA"

    def load_model(self, model_path, lora_rank, lora_alpha, quantize_to_8bit, lora_target_modules):
        print(f"--> Loading FLUX model: {model_path}"); device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": torch.bfloat16}
        if quantize_to_8bit: model_kwargs.update({"load_in_8bit": True, "device_map": {"": device}})
        if os.path.isfile(model_path):
            print("--> Detected local single file. Loading config from default repo.")
            default_repo = "black-forest-labs/FLUX.1-Kontext-dev"
            config = FluxTransformer2DModel.from_pretrained(default_repo, subfolder="transformer").config
            transformer = FluxTransformer2DModel.from_single_file(model_path, config=config, **model_kwargs)
        else:
            print("--> Detected Hugging Face repo or local folder.")
            transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer", **model_kwargs)
        target_modules = [s.strip() for s in lora_target_modules.split('\n') if s.strip()]
        print(f"--> Applying LoRA with rank={lora_rank}, alpha={lora_alpha} to modules: {target_modules}")
        config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, init_lora_weights="gaussian", target_modules=target_modules)
        if not quantize_to_8bit: transformer.to("cpu")
        transformer.add_adapter(config); print("--> LoRA Adapter applied.")
        return (transformer,)

# --------------------------------------------------------------------------------------------------

class StitchedDataPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (flux_kontext_model_files,),
                "train_image_folder": ("STRING", {"default": "C:/faceswap_training_data_stitched"}),
                "resolution": ("INT", {"default": 1024, "min": 512, "step": 64}),
                "recreate_cache": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                 "stitched_control_suffix": ("STRING", {"default": "_stitched"}),
            }
        }
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("metadata_path",); FUNCTION = "preprocess_data"; CATEGORY = "FLUX Kontext LoRA"

    def preprocess_data(self, model_path, train_image_folder, resolution, recreate_cache, stitched_control_suffix):
        print("### Starting Pre-Stitched Data Preprocessing... ###"); device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = os.path.join(train_image_folder, "kontext_lora_cache"); metadata_path = os.path.join(cache_dir, 'metadata_kontext_lora.json')
        if os.path.exists(metadata_path) and not recreate_cache:
            print(f"--> Metadata found at {metadata_path}. Skipping preprocessing."); return (metadata_path,)
        if os.path.exists(cache_dir): shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        print("--> Loading encoders and VAE..."); dtype = torch.bfloat16
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device, dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device, dtype=dtype)
        tokenizer_2 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2")
        text_encoder_2 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder_2").to(device, dtype=dtype)
        
        target_image_transforms = transforms.Compose([transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR), transforms.CenterCrop(resolution), transforms.ToTensor()])
        # The control image is stitched (wide), so we resize it to a square for the VAE.
        control_image_transforms = transforms.Compose([transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])

        image_files = glob.glob(os.path.join(train_image_folder, "*.png")) + glob.glob(os.path.join(train_image_folder, "*.jpg")) + glob.glob(os.path.join(train_image_folder, "*.webp"))
        # A target file is one that DOES NOT have the stitched suffix
        target_files = [f for f in image_files if stitched_control_suffix not in os.path.basename(f)]
        metadata = []
        
        print(f"--> Found {len(target_files)} potential target images. Processing file sets...")
        for target_path in tqdm(target_files, desc="Preprocessing Image Sets"):
            basename, ext = os.path.splitext(os.path.basename(target_path))
            
            # Find the corresponding stitched control image and caption
            stitched_control_path = os.path.join(train_image_folder, f"{basename}{stitched_control_suffix}{ext}")
            caption_path = os.path.join(train_image_folder, f"{basename}.txt")

            if not all(os.path.exists(p) for p in [stitched_control_path, caption_path]): continue

            try:
                with open(caption_path, 'r', encoding='utf-8') as f: caption = f.read().strip()
                
                # Load the target image (the "Answer Key")
                target_image = Image.open(target_path).convert("RGB")
                
                # Load the pre-stitched control image (the "Problem")
                control_image = Image.open(stitched_control_path).convert("RGB")
                
                # Convert to tensors
                target_tensor = target_image_transforms(target_image).unsqueeze(0).to(device, dtype=dtype)
                control_tensor = control_image_transforms(control_image).unsqueeze(0).to(device, dtype=dtype)
                
                with torch.no_grad():
                    train_latents = vae.encode(target_tensor).latent_dist.sample() * vae.config.scaling_factor
                    control_latents = vae.encode(control_tensor).latent_dist.sample() * vae.config.scaling_factor

                with torch.no_grad():
                    text_inputs = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                    text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
                    text_inputs_2 = tokenizer_2(caption, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt")
                    prompt_embeds_2 = text_encoder_2(text_inputs_2.input_ids.to(device)).last_hidden_state

                prompt_embeds = torch.cat([text_embeddings, prompt_embeds_2], dim=-1); pooled_prompt_embeds = prompt_embeds_2.mean(dim=1)
                cache_filepath = os.path.join(cache_dir, f"{basename}.pt")
                torch.save({"train_latents": train_latents.cpu(), "control_latents": control_latents.cpu(), "prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}, cache_filepath)
                metadata.append({"file_path": cache_filepath})

            except Exception as e: print(f"Could not process {target_path}. Error: {e}")

        with open(metadata_path, "w") as f: json.dump(metadata, f)
        print(f"--> Preprocessing complete. Saved {len(metadata)} samples. Metadata at: {metadata_path}")
        del vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2; torch.cuda.empty_cache()
        return (metadata_path,)

# --------------------------------------------------------------------------------------------------

class TrainFluxKontextLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "flux_transformer": ("MODEL",), "model_path": (flux_kontext_model_files,), "metadata_path": ("STRING", {"forceInput": True}), "output_directory": ("STRING", {"default": "C:/ComfyUI/models/loras"}), "save_name": ("STRING", {"default": "flux_face_swap_lora_v1"}), "training_steps": ("INT", {"default": 1500, "min": 1}), "learning_rate": ("FLOAT", {"default": 1.0e-4, "min": 1.0e-6, "max": 1.0e-3, "step": 1.0e-5}), "batch_size": ("INT", {"default": 1, "min": 1}), "gradient_accumulation_steps": ("INT", {"default": 4, "min": 1}), "optimizer": (["AdamW8bit", "AdamW"],), "mixed_precision": (["bf16", "fp16", "no"],), "save_every_n_steps": ("INT", {"default": 500, "min": 0}), "gradient_checkpointing": ("BOOLEAN", {"default": True}), } }
    RETURN_TYPES = (); FUNCTION = "train"; OUTPUT_NODE = True; CATEGORY = "FLUX Kontext LoRA"
    
    class KontextLoraDataset(Dataset):
        def __init__(self, metadata_path):
            with open(metadata_path, 'r') as f: self.metadata = json.load(f)
            print(f"--> Dataset initialized with {len(self.metadata)} samples.")
        def __len__(self): return len(self.metadata)
        def __getitem__(self, idx): return torch.load(self.metadata[idx]["file_path"])

    def train(self, flux_transformer, model_path, metadata_path, output_directory, save_name, training_steps, learning_rate, batch_size, gradient_accumulation_steps, optimizer, mixed_precision, save_every_n_steps, gradient_checkpointing):
        print("### Starting LoRA Training... ###"); device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}; weight_dtype = dtype_map[mixed_precision]
        flux_transformer.to(device).train()
        if gradient_checkpointing: flux_transformer.enable_gradient_checkpointing()
        params_to_optimize = [p for p in flux_transformer.parameters() if p.requires_grad]
        print(f"--> Optimizing {len(params_to_optimize)} parameters.")
        if optimizer == "AdamW8bit": optimizer_class = bnb.optim.AdamW8bit
        else: optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(params_to_optimize, lr=learning_rate)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        dataset = self.KontextLoraDataset(metadata_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        current_step = 0; pbar = tqdm(total=training_steps, desc="Training Steps"); data_iter = iter(dataloader)
        while current_step < training_steps:
            try: batch = next(data_iter)
            except StopIteration: data_iter = iter(dataloader); batch = next(data_iter)
            with torch.autocast(device_type=device, dtype=weight_dtype, enabled=(mixed_precision != "no")):
                train_latents = batch["train_latents"].to(device); control_latents = batch["control_latents"].to(device)
                prompt_embeds = batch["prompt_embeds"].to(device); pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device)
                noise = torch.randn_like(train_latents); bsz = train_latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = scheduler.add_noise(train_latents, noise, timesteps)
                conditioning_embeds = torch.cat([control_latents.view(bsz, control_latents.shape[1], -1).permute(0, 2, 1), prompt_embeds], dim=1)
                model_pred = flux_transformer(hidden_states=noisy_latents, encoder_hidden_states=conditioning_embeds, timestep=timesteps, pooled_prompt_embeds=pooled_prompt_embeds).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            (loss / gradient_accumulation_steps).backward()
            if (current_step + 1) % gradient_accumulation_steps == 0: optimizer.step(); optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"}); pbar.update(1); current_step += 1
            if save_every_n_steps > 0 and current_step % save_every_n_steps == 0: self.save_lora(flux_transformer, output_directory, f"{save_name}_step{current_step}")
        pbar.close()
        print("--> Training finished. Saving final LoRA model..."); self.save_lora(flux_transformer, output_directory, save_name)
        flux_transformer.to("cpu"); torch.cuda.empty_cache()
        return {}

    def save_lora(self, model, output_dir, file_name):
        lora_state_dict = get_peft_model_state_dict(model.to("cpu"))
        diffusers_lora_state_dict = {k.replace("base_model.model.", ""): v for k,v in lora_state_dict.items()}
        temp_save_path = os.path.join(output_dir, f"_tmp_{file_name}"); os.makedirs(temp_save_path, exist_ok=True)
        pipeline = FluxKontextPipeline(transformer=None, vae=None, text_encoder=None, tokenizer=None, scheduler=None, text_encoder_2=None, tokenizer_2=None)
        pipeline.save_lora_weights(save_directory=temp_save_path, transformer_lora_layers=diffusers_lora_state_dict)
        final_path = os.path.join(output_dir, f"{file_name}.safetensors"); original_path = os.path.join(temp_save_path, "pytorch_lora_weights.safetensors")
        if os.path.exists(original_path): shutil.copy(original_path, final_path); shutil.rmtree(temp_save_path); print(f"--> LoRA saved to: {final_path}")
        else: print(f"--> ERROR: Could not find saved LoRA file at {original_path}")

# ==================================================================================================
#                                     NODE MAPPING
# ==================================================================================================
NODE_CLASS_MAPPINGS = {
    "StitchedDataPreprocessor": StitchedDataPreprocessor,
    "LoadFluxKontextForLoRA": LoadFluxKontextForLoRA,
    "TrainFluxKontextLoRA": TrainFluxKontextLoRA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StitchedDataPreprocessor": "1. Pre-Stitched Data Preprocessor (FLUX)",
    "LoadFluxKontextForLoRA": "2. Load FLUX.1 for LoRA Training",
    "TrainFluxKontextLoRA": "3. Train FLUX.1-Kontext LoRA"
}