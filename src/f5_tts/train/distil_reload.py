# --- START OF FILE distil (10).py ---

import argparse
import os
import shutil
import math
import random
import gc
import re # Added for checkpoint loading regex

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

from importlib.resources import files
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from ema_pytorch import EMA
from safetensors.torch import load_file as load_safetensors

import torchaudio
try:
    from f5_tts.infer.utils_infer import cfg_strength as default_cfg_strength, load_vocoder, nfe_step as default_nfe_step, sway_sampling_coef as default_sway_coef
except ImportError:
    print("[Warning] f5_tts.infer.utils_infer not found. Sample logging requires manual inference parameters or defaults.")
    load_vocoder = None
    default_cfg_strength = 2.0
    default_nfe_step = 32
    default_sway_coef = None

try:
    from f5_tts.model import CFM, UNetT, DiT, Trainer # Trainer only for potential utils
    from f5_tts.model.utils import get_tokenizer, seed_everything, default, exists, lens_to_mask, list_str_to_idx, list_str_to_tensor, mask_from_frac_lengths
    from f5_tts.model.dataset import load_dataset, DynamicBatchSampler, collate_fn
    from f5_tts.model.modules import MelSpec
except ImportError as e:
    print(f"Error importing F5-TTS components: {e}")
    print("Please ensure the script is run from a directory where 'f5_tts' is importable, or the package is installed.")
    exit(1)

from f5_tts.model.duration_predictor import DurationPredictor

# -------------------------- Constants & Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
SEED = 666

# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Distillation Finetuning for F5-TTS")
    parser.add_argument("--teacher_ckpt_path", type=str, required=True, help="Path to the full teacher model checkpoint (.pt or .safetensors)")
    parser.add_argument("--student_exp_name", type=str, default="F5TTS_v1_Custom_Prune_12", choices=["F5TTS_v1_Custom_Prune_14", "F5TTS_v1_Custom_Prune_12"], help="Experiment name identifying the STUDENT architecture")
    parser.add_argument("--student_init_ckpt_path", type=str, default=None, help="Path to the initial student checkpoint to start distillation from. If None, student starts from scratch or random init.")
    parser.add_argument("--dataset_name", type=str, default="Emilia_ZH_EN", help="Name of the dataset to use")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save distillation checkpoints. Defaults to ckpts/{dataset_name}_distill_{student_exp_name}")
    parser.add_argument("--distill_loss_weight", type=float, default=0.5, help="Weight (alpha) for the distillation loss.")
    parser.add_argument(
            "--spec_l1_weight", type=float, default=0.0,
            help="Weight (β) for an additional L1 loss on the final mel outputs (for sharper transients)."
        )
    parser.add_argument("--distill_loss_type", type=str, default="mse", choices=["mse", "l1"], help="Type of loss for comparing teacher/student flow predictions.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for STUDENT training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for STUDENT optimizer")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4800, help="Batch size per GPU (frames or samples)")
    parser.add_argument("--batch_size_type", type=str, default="frame", choices=["frame", "sample"], help="Batch size type")
    parser.add_argument("--max_samples", type=int, default=64, help="Max sequences per batch (if batch_size_type='frame')")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--epochs", type=int, default=50, help="Number of distillation epochs")
    parser.add_argument("--num_warmup_updates", type=int, default=10000, help="Warmup updates for LR scheduler")
    parser.add_argument("--save_per_updates", type=int, default=25000, help="Save checkpoint every N updates")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=5, help="-1 to keep all, 0 to not save intermediate, > 0 to keep last N")
    parser.add_argument("--last_per_updates", type=int, default=5000, help="Save last checkpoint every N updates")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=[None, "wandb", "tensorboard"], help="Logger type") # Defaulted to tensorboard
    parser.add_argument("--logging_dir", type=str, default="runs", help="Base directory for TensorBoard logs (if logger='tensorboard')")
    parser.add_argument("--bnb_optimizer", action="store_true", help="Use 8-bit Adam optimizer")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use Exponential Moving Average for student model")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="Decay rate for EMA")
    parser.add_argument("--tokenizer", type=str, default="char", choices=["pinyin", "char", "custom"], help="Tokenizer type")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to custom tokenizer vocab file")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--log_samples", action="store_true", help="Log inferenced audio samples during training.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no","fp16","bf16"],
                        help="Mixed precision mode for Accelerate")
    parser.add_argument("--use_duration_predictor", action="store_true", default=False, 
                        help="Enable training of duration predictor during distillation")
    parser.add_argument("--duration_loss_weight", type=float, default=None, 
                        help="Weight for duration prediction loss")    
    parser.add_argument("--resume_epoch", type=float, default=None,
                        help="Resume from specific epoch")    
    parser.add_argument("--from_scratch", action="store_true", default=False,
                        help="From scratch")    

    parser.add_argument("--ref_texts", type=str, nargs="+", default=None, help="Reference texts for sample generation")
    parser.add_argument("--ref_audio_paths", type=str, nargs="+", default=None, help="Paths to reference audio files")
    parser.add_argument("--ref_sample_text_prompts", type=str, nargs="+", default=None, help="Text prompts for reference samples")
    
    
    return parser.parse_args()

# -------------------------- Helper Functions --------------------------- #

def load_model_checkpoint(model, ckpt_path, device, accelerator):
    """
    Loads state dict from .pt or .safetensors into the model.
    Handles prefix stripping and adds 'transformer.' if needed.
    Loads tensors to the specified device (CPU or GPU).
    """
    if not os.path.exists(ckpt_path):
        accelerator.print(f"Checkpoint path not found: {ckpt_path}")
        return False

    accelerator.print(f"Loading checkpoint: {ckpt_path}")
    state_dict_raw = None
    checkpoint_data = None

    try:
        if ckpt_path.endswith(".safetensors"):
             try: state_dict_raw = load_safetensors(ckpt_path, device=str(device))
             except Exception:
                  accelerator.print(f"Direct load to {device} failed, loading to CPU first.")
                  state_dict_raw = load_safetensors(ckpt_path, device="cpu")
             accelerator.print("Loaded state_dict from .safetensors file.")
        elif ckpt_path.endswith(".pt"):
             accelerator.print("Loading .pt file with weights_only=False. Ensure checkpoint source is trusted.")
             checkpoint_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
             accelerator.print("Loaded checkpoint from .pt file.")
             keys_to_try = ['model_state_dict', 'state_dict', 'ema_model_state_dict', 'model']
             for key in keys_to_try:
                 if key in checkpoint_data and isinstance(checkpoint_data[key], dict):
                     state_dict_raw = checkpoint_data[key]; accelerator.print(f"Found model weights under key: '{key}'"); break
             if state_dict_raw is None:
                 if isinstance(checkpoint_data, dict): state_dict_raw = checkpoint_data; accelerator.print("Assuming the loaded .pt file root is the state_dict.")
                 else: raise ValueError("Could not find a valid state_dict within the .pt file.")
        else: raise ValueError(f"Unsupported checkpoint file extension: {ckpt_path}")

        if not isinstance(state_dict_raw, dict) or not state_dict_raw: raise ValueError("Loaded state_dict is not a valid dictionary or is empty.")

        cleaned_state_dict = {}
        prefixes_to_strip = ["module.", "_orig_mod.", "ema_model."]
        is_transformer_state_dict = all(not k.startswith("transformer.") for k in state_dict_raw.keys()) and \
                                    any("attn" in k or "ff." in k or "embed" in k for k in state_dict_raw.keys())

        prefix_to_strip = None
        first_key = next(iter(state_dict_raw.keys()), None)
        if first_key:
            for p in prefixes_to_strip:
                if first_key.startswith(p) and sum(1 for k in state_dict_raw if k.startswith(p)) > 0.8 * len(state_dict_raw):
                     prefix_to_strip = p; break

        if prefix_to_strip:
             accelerator.print(f"Stripping prefix '{prefix_to_strip}' from state_dict keys.")
             prefix_len = len(prefix_to_strip)
             for k, v in state_dict_raw.items():
                  cleaned_state_dict[k[prefix_len:] if k.startswith(prefix_to_strip) else k] = v
        else:
             accelerator.print("No common prefix detected or necessary to strip.")
             cleaned_state_dict = state_dict_raw

        if hasattr(model, "transformer") and isinstance(model.transformer, nn.Module) and is_transformer_state_dict:
            accelerator.print("Detected transformer-only state_dict. Adding 'transformer.' prefix.")
            prefixed_state_dict = {}
            for k, v in cleaned_state_dict.items():
                prefixed_state_dict[f"transformer.{k}" if not k.startswith("transformer.") else k] = v
            cleaned_state_dict = prefixed_state_dict

        final_state_dict = {}
        ignore_keys = {"initted", "step"}
        for k, v in cleaned_state_dict.items():
            if k not in ignore_keys:
                final_state_dict[k] = v.to(device) # Move tensor to target device here

        if not final_state_dict: accelerator.print("ERROR: State dictionary became empty."); return False

        accelerator.print(f"Attempting to load state_dict into the full '{type(model).__name__}' model...")
        incompatible_keys = model.load_state_dict(final_state_dict, strict=False)
        accelerator.print("Loaded state dict into model.")
        if incompatible_keys.missing_keys: accelerator.print(f"Note: Missing keys: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys: accelerator.print(f"Note: Unexpected keys: {incompatible_keys.unexpected_keys}")

        del state_dict_raw, cleaned_state_dict, final_state_dict
        if checkpoint_data is not None: del checkpoint_data
        gc.collect()
        if isinstance(device, torch.device) and device.type == 'cuda': torch.cuda.empty_cache()
        elif isinstance(device, torch.device) and device.type == 'xpu': torch.xpu.empty_cache()
        return True
    except Exception as e:
        accelerator.print(f"Error loading checkpoint file {ckpt_path}: {e}")
        if 'state_dict_raw' in locals(): del state_dict_raw
        if 'cleaned_state_dict' in locals(): del cleaned_state_dict
        if 'final_state_dict' in locals(): del final_state_dict
        if 'checkpoint_data' in locals(): del checkpoint_data
        gc.collect()
        if isinstance(device, torch.device) and device.type == 'cuda': torch.cuda.empty_cache()
        elif isinstance(device, torch.device) and device.type == 'xpu': torch.xpu.empty_cache()
        return False


def generate_reference_samples(
    accelerator, 
    student_model, 
    ema_model, 
    global_update, 
    vocoder, 
    log_samples_path,
    ref_texts,
    ref_mels,
    ref_sample_text_prompts,
    nfe_step,
    cfg_strength,
    sway_sampling_coef
):
    """
    Generate samples from fixed reference examples for consistent quality monitoring.
    Uses combined Ref+Prompt CHARACTER text and slices output mel.
    
    Args:
        accelerator: The Accelerator instance
        student_model: The student model (wrapped by accelerator)
        ema_model: The EMA model (if available)
        global_update: Current update step
        vocoder: The vocoder to convert mel spectrograms to audio
        log_samples_path: Path to save sample outputs
        ref_texts: List of reference texts
        ref_mels: List of reference mel spectrograms
        ref_sample_text_prompts: List of text prompts to generate
        nfe_step: Number of flow steps for sampling
        cfg_strength: Classifier-free guidance strength
        sway_sampling_coef: Sway sampling coefficient
    """
    # Ensure this runs only on the main process and if references are enabled
    if not accelerator.is_main_process or not ref_mels:
        return
    
    import os
    import torch
    import torchaudio
    
    os.makedirs(log_samples_path, exist_ok=True)
    
    # Select the model for sampling (EMA preferably)
    unwrapped_model = accelerator.unwrap_model(student_model)
    model_for_sampling = ema_model.ema_model if ema_model is not None else unwrapped_model
    target_sample_rate = unwrapped_model.mel_spec.target_sample_rate
    
    # Set model to evaluation mode for sampling
    model_for_sampling.eval() 
    
    with torch.inference_mode():
        # Iterate through each reference mel provided
        for idx, ref_mel in enumerate(ref_mels):
            try:
                # --- Get Corresponding Reference Text and Prompt Text ---
                ref_text = ref_texts[idx] if idx < len(ref_texts) else ""
                prompt_text_for_generation = ref_sample_text_prompts[idx] if idx < len(ref_sample_text_prompts) else "Default test reference."
                original_full_prompt = prompt_text_for_generation # For saving later
                
                # --- Combine text using CHARACTERS ---
                # Ensure texts are strings. Add space if needed for separation.
                combined_text = str(ref_text) + " " + str(prompt_text_for_generation)
                # Wrap in a list as model.sample expects list[str] or tensor
                final_text_list_for_model = [combined_text]
                accelerator.print(f"Using combined CHARACTER text for model: {combined_text[:100]}...")

                # Prepare the reference mel spectrogram
                ref_mel = ref_mel.to(accelerator.device)
                ref_mel_for_sample = ref_mel.permute(0, 2, 1) # Shape [1, mel_len, n_mel_channels]

                # Decode and Save Original Reference Audio
                accelerator.print(f"Decoding reference mel (shape: {ref_mel.shape}) for saving source audio")
                if hasattr(vocoder, 'decode'):
                    ref_audio = vocoder.decode(ref_mel).cpu()
                else:
                    ref_audio = vocoder(ref_mel).squeeze(0).cpu()
                    
                ref_path = f"{log_samples_path}/update_{global_update}_ref{idx}_source.wav"
                if ref_audio.ndim == 1: 
                    ref_audio = ref_audio.unsqueeze(0)
                elif ref_audio.ndim == 3 and ref_audio.shape[1] == 1: 
                    ref_audio = ref_audio.squeeze(1)
                torchaudio.save(ref_path, ref_audio, target_sample_rate)
                accelerator.print(f"Saved reference audio source: {ref_path}")
                
                # Determine Target Duration
                batch_size, cond_seq_len = ref_mel_for_sample.shape[:2]
                target_duration_frames = cond_seq_len * 2 
                target_duration = torch.full((batch_size,), target_duration_frames, device=ref_mel_for_sample.device, dtype=torch.long)
                accelerator.print(f"Requesting target duration for sample: {target_duration.item()} frames")

                # Generate Mel Spectrogram using the Model with CHARACTER text
                accelerator.print(f"Generating sample mel spectrogram...")
                generated_mel, _ = model_for_sampling.sample(
                    cond=ref_mel_for_sample,
                    text=final_text_list_for_model,
                    duration=target_duration, 
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )
                
                # Check for Invalid Values
                if torch.isnan(generated_mel).any() or torch.isinf(generated_mel).any():
                    accelerator.print(f"Error: NaNs or Infs detected in generated mel for reference sample {idx}. Skipping generation.")
                    continue 
                    
                # Prepare Generated Mel for Vocoder
                generated_mel = generated_mel.to(torch.float32)
                ref_mel_frames = ref_mel.shape[-1] # Same as cond_seq_len
                accelerator.print(f"Original reference mel frames: {ref_mel_frames}")
                accelerator.print(f"Generated mel shape: {generated_mel.shape}")

                # Slice the output mel like the Wrapper
                if generated_mel.shape[1] > ref_mel_frames:
                     mel_for_vocoder = generated_mel[:, ref_mel_frames:, :].permute(0, 2, 1).to(accelerator.device)
                     accelerator.print(f"Using SLICED generated mel (after ref) for vocoder. Shape: {mel_for_vocoder.shape}")
                else:
                     accelerator.print(f"Warning: Generated mel length ({generated_mel.shape[1]}) not longer than reference ({ref_mel_frames}). Output slice is empty.")
                     mel_for_vocoder = None 

                # Decode Sliced Mel to Audio
                if mel_for_vocoder is not None and mel_for_vocoder.shape[-1] > 0:
                     accelerator.print(f"Decoding sliced mel for saving generated audio")
                     if hasattr(vocoder, 'decode'):
                         gen_audio = vocoder.decode(mel_for_vocoder).cpu()
                     else:
                         gen_audio = vocoder(mel_for_vocoder).squeeze(0).cpu()

                     # Ensure audio tensor shape [1, audio_len] for torchaudio.save
                     if gen_audio.ndim == 1: 
                         gen_audio = gen_audio.unsqueeze(0)
                     elif gen_audio.ndim == 3 and gen_audio.shape[1] == 1: 
                         gen_audio = gen_audio.squeeze(1)

                     # Save Generated Audio
                     gen_path = f"{log_samples_path}/update_{global_update}_ref{idx}_gen.wav"
                     torchaudio.save(gen_path, gen_audio, target_sample_rate)
                     accelerator.print(f"Saved generated sample (from sliced mel): {gen_path}")
                else:
                     accelerator.print("Skipping saving generated audio: sliced mel part is empty or generation too short.")

                # Save Prompt Text (Save the original prompt for clarity)
                txt_path = f"{log_samples_path}/update_{global_update}_ref{idx}_prompt.txt"
                with open(txt_path, 'w') as f:
                    f.write(original_full_prompt)
                
                accelerator.print(f"Completed processing for reference sample {idx} at update {global_update}")
                
            except Exception as e:
                accelerator.print(f"Error processing reference sample {idx}: {e}")
                import traceback
                traceback.print_exc()
    
    # Restore model to training mode after processing all samples
    model_for_sampling.train()

def preload_reference_mels(accelerator, ref_audio_paths, mel_spec_module):
    """
    Pre-loads reference mel spectrograms from the provided audio paths.
    
    Args:
        accelerator: The Accelerator instance
        ref_audio_paths: List of paths to reference audio files
        mel_spec_module: The mel spectrogram module to use for conversion
    
    Returns:
        List of pre-loaded mel spectrograms
    """
    if not accelerator.is_main_process or not ref_audio_paths:
        return []
    
    import os
    import torch
    import torchaudio
    
    ref_mels = []
    target_sample_rate = mel_spec_module.target_sample_rate
    
    accelerator.print(f"Loading reference audios with target sample rate: {target_sample_rate}")
    for audio_path in ref_audio_paths:
        if os.path.exists(audio_path):
            # Load audio and convert to mel spectrogram
            accelerator.print(f"Loading reference audio: {audio_path}")
            waveform, sr = torchaudio.load(audio_path)
            accelerator.print(f"  Original sample rate: {sr}")
            
            if sr != target_sample_rate:
                accelerator.print(f"  Resampling from {sr} to {target_sample_rate}")
                waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
            
            # Convert to mel spectrogram
            mel = mel_spec_module(waveform.to(accelerator.device)).cpu()
            ref_mels.append(mel)
            accelerator.print(f"  Successfully loaded reference audio. Mel shape: {mel.shape}")
        else:
            accelerator.print(f"Warning: Reference audio file not found: {audio_path}")
            
    return ref_mels

# Example usage in main() function:
"""
# In parse_args(), add:
parser.add_argument("--ref_texts", type=str, nargs="+", default=None, help="Reference texts for sample generation")
parser.add_argument("--ref_audio_paths", type=str, nargs="+", default=None, help="Paths to reference audio files")
parser.add_argument("--ref_sample_text_prompts", type=str, nargs="+", default=None, help="Text prompts for reference samples")

# In main() function, after initializing mel_spec_module:
ref_texts = args.ref_texts or []
ref_audio_paths = args.ref_audio_paths or []
ref_sample_text_prompts = args.ref_sample_text_prompts or []

# After initializing mel_spec_module:
ref_mels = []
if accelerator.is_main_process and args.log_samples and ref_audio_paths:
    ref_mels = preload_reference_mels(accelerator, ref_audio_paths, mel_spec_module)
    
# In sample logging section where vocoder is loaded:
if args.log_samples and accelerator.is_main_process and ref_mels:
    # Generate reference samples
    generate_reference_samples(
        accelerator, student_model, ema_model, global_update, vocoder, 
        log_samples_path, ref_texts, ref_mels, ref_sample_text_prompts,
        nfe_step_sample, cfg_strength_sample, sway_sampling_coef_sample
    )
"""

def save_distill_checkpoint(accelerator, student_model, optimizer, scheduler, ema_model, update, epoch, checkpoint_path, keep_last_n, is_last=False):
    """
    Saves the distillation checkpoint using accelerator.save.
    Operates on the *prepared* objects.
    Handles checkpoint cleanup.
    """
    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)

        checkpoint = dict(
            # Save state dict from the *unwrapped* model to remove DDP wrapper
            model_state_dict=accelerator.unwrap_model(student_model).state_dict(),
            optimizer_state_dict=optimizer.state_dict(), # Save state dict from prepared optimizer
            scheduler_state_dict=scheduler.state_dict(), # Save state dict from prepared scheduler
            update=update,
            epoch=epoch,  # Added this line to save the epoch
        )
        if ema_model is not None:
            # EMA state is not handled by accelerator, save its state dict directly
            checkpoint['ema_model_state_dict'] = ema_model.state_dict()

        ckpt_name = f"model_{'last' if is_last else update}.pt"
        ckpt_file = os.path.join(checkpoint_path, ckpt_name)
        accelerator.save(checkpoint, ckpt_file) # Use accelerator.save for robust saving
        print(f"Saved {'last ' if is_last else ''}student checkpoint: {ckpt_file}")

        # Checkpoint Cleanup
        if not is_last and keep_last_n != 0: # Allow keep_last_n = -1 (keep all)
            try:
                checkpoints = [ f for f in os.listdir(checkpoint_path) if f.startswith("model_") and f.endswith(".pt") and f != "model_last.pt" ]
                numeric_checkpoints = []
                for ckpt in checkpoints:
                    match = re.search(r'model_(\d+)\.pt', ckpt)
                    if match: numeric_checkpoints.append((int(match.group(1)), ckpt))
                    else: print(f"Skipping non-numeric checkpoint during cleanup: {ckpt}")

                numeric_checkpoints.sort(key=lambda x: x[0]) # Sort by update number (ascending)
                sorted_ckpt_files = [ckpt_file for _, ckpt_file in numeric_checkpoints]

                if keep_last_n > 0: # Only remove if keep_last_n is positive
                    while len(sorted_ckpt_files) > keep_last_n:
                        oldest_checkpoint = sorted_ckpt_files.pop(0)
                        try: os.remove(os.path.join(checkpoint_path, oldest_checkpoint)); print(f"Removed old student checkpoint: {oldest_checkpoint}")
                        except OSError as e: print(f"Warning: Error removing old checkpoint {oldest_checkpoint}: {e}")
            except Exception as e:
                print(f"Warning: Error during checkpoint cleanup: {e}")


def load_distill_checkpoint(accelerator, student_model, optimizer, scheduler, ema_model, checkpoint_path):
    """
    Loads the student model's state to resume distillation.
    Loads state dicts directly into the passed (unprepared) objects using torch.load.
    Returns the starting update number and epoch.
    """
    start_update = 0
    start_epoch = 0
    latest_checkpoint_path = None

    if exists(checkpoint_path) and os.path.isdir(checkpoint_path):
        last_ckpt_path = os.path.join(checkpoint_path, "model_last.pt")
        if os.path.exists(last_ckpt_path):
            latest_checkpoint_path = last_ckpt_path
            accelerator.print(f"Found last student checkpoint: {latest_checkpoint_path}")
        else:
            checkpoints = [ f for f in os.listdir(checkpoint_path) if f.startswith("model_") and f.endswith(".pt") and f != "model_last.pt" ]
            if checkpoints:
                numeric_checkpoints = []
                for ckpt in checkpoints:
                    match = re.search(r'model_(\d+)\.pt', ckpt)
                    if match: numeric_checkpoints.append((int(match.group(1)), ckpt))
                if numeric_checkpoints:
                     numeric_checkpoints.sort(key=lambda x: x[0], reverse=True) # Sort descending by update number
                     latest_checkpoint_path = os.path.join(checkpoint_path, numeric_checkpoints[0][1])
                     accelerator.print(f"Found latest training student checkpoint: {latest_checkpoint_path}")
                else: accelerator.print("No numbered checkpoints found to resume from.")

    if latest_checkpoint_path is None or not os.path.exists(latest_checkpoint_path):
        accelerator.print("No valid student checkpoint found for resuming distillation.")
        return 0, 0 # Return 0 updates processed and epoch 0

    accelerator.print(f"Loading distillation state from: {latest_checkpoint_path} (before prepare)")
    
    # Track what was successfully loaded
    model_loaded = False
    optimizer_loaded = False
    scheduler_loaded = False
    ema_loaded = True if ema_model is None else False
    
    try:
        # Load to CPU first using torch.load as objects are not prepared yet
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
        
        # --- Load model state (independent of other components) ---
        model_sd = checkpoint.get('model_state_dict')
        if model_sd:
            try:
                incompatible_keys = student_model.load_state_dict(model_sd, strict=False) # Load into raw model
                model_loaded = True
                accelerator.print("Loaded student model state (raw).")
                if incompatible_keys.missing_keys: accelerator.print(f" Missing keys: {incompatible_keys.missing_keys}")
                if incompatible_keys.unexpected_keys: accelerator.print(f" Unexpected keys: {incompatible_keys.unexpected_keys}")
            except Exception as e:
                accelerator.print(f"Error loading model state: {e}")
        else: 
            accelerator.print("Warning: model_state_dict not found in checkpoint.")

        # --- Load optimizer state (only if model was loaded) ---
        if model_loaded:
            opt_sd = checkpoint.get('optimizer_state_dict')
            if opt_sd:
                try:
                    optimizer.load_state_dict(opt_sd) # Load into raw optimizer
                    optimizer_loaded = True
                    accelerator.print("Loaded optimizer state (raw).")
                except Exception as e:
                    accelerator.print(f"Warning: Could not load optimizer state: {e}")
                    accelerator.print("Continuing with fresh optimizer but keeping model weights")
            else: 
                accelerator.print("Warning: optimizer_state_dict not found.")

            # --- Load scheduler state (only if optimizer was loaded) ---
            if optimizer_loaded:
                sched_sd = checkpoint.get('scheduler_state_dict')
                if sched_sd:
                    try:
                        scheduler.load_state_dict(sched_sd) # Load into raw scheduler
                        scheduler_loaded = True
                        accelerator.print("Loaded scheduler state (raw).")
                    except Exception as e:
                        accelerator.print(f"Warning: Could not load scheduler state: {e}")
                        accelerator.print("Continuing with fresh scheduler")
                else: 
                    accelerator.print("Warning: scheduler_state_dict not found.")

            # --- Load EMA state (if applicable, independent of other states) ---
            if ema_model is not None:
                ema_sd = checkpoint.get('ema_model_state_dict')
                if ema_sd:
                    try:
                        ema_model.load_state_dict(ema_sd) # Load into raw EMA object
                        ema_loaded = True
                        accelerator.print("Loaded EMA state (raw).")
                    except Exception as e: 
                        accelerator.print(f"Warning: Could not load EMA state: {e}. EMA will start fresh.")
                else: 
                    accelerator.print("Warning: ema_model_state_dict not found. EMA will start fresh.")
        
        # --- Determine update and epoch ---
        if model_loaded:
            # Get update from checkpoint if available
            update_resumed_from = checkpoint.get('update', 0)
            start_update = update_resumed_from + 1 if optimizer_loaded else 0
            
            # If optimizer couldn't be loaded, we'll start from update 0
            if not optimizer_loaded:
                accelerator.print("Since optimizer couldn't be loaded, starting from update 0 with loaded model weights")
            
            # Check for epoch in checkpoint
            start_epoch = checkpoint.get('epoch', 0)
            
            if optimizer_loaded:
                accelerator.print(f"Resuming distillation from update {start_update}, epoch {start_epoch+1}")
            else:
                accelerator.print(f"Starting from update 0 with pre-trained model weights")
        else:
            # If model couldn't be loaded, start completely fresh
            accelerator.print("Could not load model state. Starting completely from scratch.")
            start_update = 0
            start_epoch = 0

        # Cleanup
        del checkpoint
        if 'model_sd' in locals(): del model_sd
        if 'opt_sd' in locals(): del opt_sd
        if 'sched_sd' in locals(): del sched_sd
        if ema_model is not None and 'ema_sd' in locals(): del ema_sd
        gc.collect()
        
        # Return appropriate values based on what was loaded
        return start_update, start_epoch

    except Exception as e:
        accelerator.print(f"Error handling checkpoint: {e}. Starting from scratch.")
        if 'checkpoint' in locals(): del checkpoint
        if 'model_sd' in locals(): del model_sd
        if 'opt_sd' in locals(): del opt_sd
        if 'sched_sd' in locals(): del sched_sd
        if ema_model is not None and 'ema_sd' in locals(): del ema_sd
        gc.collect()
        return 0, 0 # Start from scratch, epoch 0
        
# -------------------------- Main Distillation Function --------------------------- #

def main():
    args = parse_args()
    seed_everything(SEED)

    # --- Accelerator Setup ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    # Handle logger explicitly later based on imports
    accelerator = Accelerator(
        log_with=None, # Initialize logger manually later
        gradient_accumulation_steps=args.grad_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.even_batches = False # Crucial for DynamicBatchSampler
    accelerator.print(f"Using {accelerator.num_processes} GPUs.")
    accelerator.print(f"Distillation Args: {args}")

    # --- Paths ---
    try:
        base_module_path = files("f5_tts")
        base_data_path = str(base_module_path.joinpath("../../data"))
        base_ckpt_path = str(base_module_path.joinpath("../../ckpts"))
        if not os.path.isdir(base_data_path): raise FileNotFoundError("Data path not found")
        if not os.path.isdir(base_ckpt_path): raise FileNotFoundError("Ckpt path not found")
    except Exception as e:
        accelerator.print(f"[Warning] Error determining base paths: {e}. Using fallback './data', './ckpts'.")
        base_data_path = "./data"; base_ckpt_path = "./ckpts"
    accelerator.print(f"Using base data path: {base_data_path}")
    accelerator.print(f"Using base ckpt path: {base_ckpt_path}")

    # --- Output Directory ---
    if args.output_dir is None:
        args.output_dir = os.path.join(base_ckpt_path, f"{args.dataset_name}_distill_{args.student_exp_name}")
    if accelerator.is_main_process: os.makedirs(args.output_dir, exist_ok=True)
    accelerator.print(f"Distillation checkpoints will be saved to: {args.output_dir}")

    # --- Tokenizer ---
    tokenizer_type_for_init = args.tokenizer
    path_or_alias_for_get_tokenizer = args.dataset_name
    if args.student_exp_name in ["F5TTS_v1_Custom_Prune_14", "F5TTS_v1_Custom_Prune_12"]:
        extended_vocab_file_path = os.path.join(base_data_path, f"{args.dataset_name}_{args.tokenizer}", "vocab.txt")
        fallback_path = os.path.join(base_data_path, args.dataset_name, "vocab.txt")
        accelerator.print(f"Checking for extended vocab at: {extended_vocab_file_path} and fallback {fallback_path}")
        if os.path.isfile(extended_vocab_file_path): path_or_alias_for_get_tokenizer = extended_vocab_file_path; tokenizer_type_for_init = "custom"
        elif os.path.isfile(fallback_path): path_or_alias_for_get_tokenizer = fallback_path; tokenizer_type_for_init = "custom"
        else: accelerator.print(f"[Warning] Extended vocab not found. Using default dataset alias '{args.dataset_name}'.")
    elif args.tokenizer == "custom":
        if not args.tokenizer_path or not os.path.isfile(args.tokenizer_path): accelerator.print(f"ERROR: Custom tokenizer path '{args.tokenizer_path}' invalid."); exit(1)
        path_or_alias_for_get_tokenizer = args.tokenizer_path; tokenizer_type_for_init = "custom"

    accelerator.print(f"Using tokenizer source: '{path_or_alias_for_get_tokenizer}', type: '{tokenizer_type_for_init}'")
    try:
        vocab_char_map, vocab_size = get_tokenizer(path_or_alias_for_get_tokenizer, tokenizer_type_for_init)
        accelerator.print(f"Tokenizer loaded. Vocab Size: {vocab_size}")
    except Exception as e: accelerator.print(f"ERROR loading tokenizer: {e}"); exit(1)

    # --- Mel Spectrogram Config ---
    mel_spec_kwargs = dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mel_channels=n_mel_channels, target_sample_rate=target_sample_rate, mel_spec_type=mel_spec_type)
    mel_spec_module = MelSpec(**mel_spec_kwargs)

    # --- Reference Samples Setup ---
    ref_texts = args.ref_texts or []
    ref_audio_paths = args.ref_audio_paths or []
    ref_sample_text_prompts = args.ref_sample_text_prompts or []
    ref_mels = []
    
    # Ensure lists match in length or provide warnings
    max_len = max(len(ref_texts), len(ref_audio_paths), len(ref_sample_text_prompts))
    if max_len > 0:
        if len(ref_texts) > 0 and len(ref_texts) != max_len:
            accelerator.print(f"Warning: ref_texts length ({len(ref_texts)}) does not match other reference lists ({max_len}). Using available items only.")
        if len(ref_audio_paths) > 0 and len(ref_audio_paths) != max_len:
            accelerator.print(f"Warning: ref_audio_paths length ({len(ref_audio_paths)}) does not match other reference lists ({max_len}). Using available items only.")
        if len(ref_sample_text_prompts) > 0 and len(ref_sample_text_prompts) != max_len:
            accelerator.print(f"Warning: ref_sample_text_prompts length ({len(ref_sample_text_prompts)}) does not match other reference lists ({max_len}). Using available items only.")

    # --- Model Configuration ---
    teacher_model_cls = DiT; teacher_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    student_model_cls = DiT; student_model_cfg = None
    if args.student_exp_name == "F5TTS_v1_Custom_Prune_14": student_model_cfg = dict(dim=1024, depth=14, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    elif args.student_exp_name == "F5TTS_v1_Custom_Prune_12": student_model_cfg = dict(dim=1024, depth=12, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    else: accelerator.print(f"ERROR: Unsupported student experiment name: {args.student_exp_name}"); exit(1)

    # --- Instantiate Models (Raw, on CPU initially) ---
    accelerator.print("Initializing Teacher Model (CPU)...")
    teacher_model = CFM(transformer=teacher_model_cls(**teacher_model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels), mel_spec_module=mel_spec_module, vocab_char_map=vocab_char_map)
    accelerator.print("Initializing Student Model (CPU)...")
    student_model = CFM(transformer=student_model_cls(**student_model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels), mel_spec_module=mel_spec_module, vocab_char_map=vocab_char_map)

    # STEVE: Duration Predictor - all process, not only main process
    # Only add duration predictor if the flag is set
    if args.use_duration_predictor:
        from f5_tts.model.duration_predictor import DurationPredictor
        student_duration_predictor = DurationPredictor(vocab_size, 512, 32, 3, 0.5)
        setattr(student_model, 'duration_predictor', student_duration_predictor)
        accelerator.print("Duration predictor added to student model")
    else:
        # Remove duration predictor if it exists
        if hasattr(student_model, 'duration_predictor'):
            delattr(student_model, 'duration_predictor')
            accelerator.print("Duration predictor removed from student model")
            
    # --- Load Teacher Checkpoint (to accelerator.device) ---
    accelerator.print(f"Loading Teacher checkpoint: {args.teacher_ckpt_path}")
    teacher_load_success = load_model_checkpoint(teacher_model, args.teacher_ckpt_path, accelerator.device, accelerator)
    if not teacher_load_success: accelerator.print("Failed to load teacher checkpoint. Exiting."); return
    teacher_model.eval(); teacher_model.to(accelerator.device) # Ensure on device
    for param in teacher_model.parameters(): param.requires_grad = False
    accelerator.print("Teacher model loaded, frozen, and moved to device.")

    # --- Prepare Initial Student Checkpoint Path (Copy if needed) ---
    initial_student_ckpt_path_for_loading = args.student_init_ckpt_path
    if (not args.from_scratch) and args.student_init_ckpt_path and os.path.exists(args.student_init_ckpt_path):
        if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
        source_ckpt_basename = os.path.basename(args.student_init_ckpt_path)
        target_basename = "pretrained_" + source_ckpt_basename if not source_ckpt_basename.startswith("pretrained_") else source_ckpt_basename
        copied_checkpoint_path = os.path.join(args.output_dir, target_basename)
        if not os.path.isfile(copied_checkpoint_path):
            shutil.copy(args.student_init_ckpt_path, copied_checkpoint_path)

        '''
        # Check if the file exists and verify it's valid
        if os.path.isfile(copied_checkpoint_path):
            try:
                # Test if file is valid by attempting to read header
                with open(copied_checkpoint_path, 'rb') as f:
                    # Just read a small chunk to see if file is accessible
                    header = f.read(8)
                accelerator.print(f"Using existing copied checkpoint: {copied_checkpoint_path}")
                initial_student_ckpt_path_for_loading = copied_checkpoint_path
            except Exception as e:
                accelerator.print(f"Warning: Existing checkpoint appears corrupted: {e}. Will create fresh copy.")
                try:
                    os.remove(copied_checkpoint_path)
                except Exception:
                    pass
                # Force recopy below
                copied_checkpoint_path = None
        
        # Copy file using block-by-block approach to avoid memory issues
        if not os.path.isfile(copied_checkpoint_path):
            accelerator.print(f"Copying initial student checkpoint to: {copied_checkpoint_path}")
            try:
                # More robust file copying with block reading/writing
                with open(args.student_init_ckpt_path, 'rb') as src_file:
                    with open(copied_checkpoint_path, 'wb') as dst_file:
                        # Copy in chunks of 10MB
                        chunk_size = 10 * 1024 * 1024  
                        while True:
                            chunk = src_file.read(chunk_size)
                            if not chunk:
                                break
                            dst_file.write(chunk)
                
                # Verify copied file size matches source
                src_size = os.path.getsize(args.student_init_ckpt_path)
                dst_size = os.path.getsize(copied_checkpoint_path)
                
                if src_size == dst_size:
                    accelerator.print(f"Successfully copied checkpoint ({dst_size} bytes)")
                    initial_student_ckpt_path_for_loading = copied_checkpoint_path
                else:
                    accelerator.print(f"Warning: Copied file size mismatch ({src_size} vs {dst_size}). Using original path.")
                    initial_student_ckpt_path_for_loading = args.student_init_ckpt_path
            except Exception as e:
                accelerator.print(f"Warning: Error copying checkpoint: {e}. Using original path.")
                initial_student_ckpt_path_for_loading = args.student_init_ckpt_path
    '''
    # --- Load Initial Student State (if provided, into raw model on CPU) ---
    if (not args.from_scratch) and initial_student_ckpt_path_for_loading:
        accelerator.print(f"Loading initial Student state from: {initial_student_ckpt_path_for_loading} (before prepare)")
        student_load_success = load_model_checkpoint(student_model, initial_student_ckpt_path_for_loading, torch.device("cpu"), accelerator)
        if not student_load_success: accelerator.print("[Warning] Failed to load initial student checkpoint. Starting fresh.")
        else: accelerator.print("Initial student state loaded successfully (CPU)."); gc.collect()
    else: accelerator.print("No initial student checkpoint provided. Starting fresh.")

    # --- Check for Duration Predictor and Add if Missing ---
    accelerator.print("Checking for duration predictor in student model...")
    has_duration_predictor = hasattr(student_model, 'duration_predictor') and student_model.duration_predictor is not None
    
    if not has_duration_predictor:
        accelerator.print("Duration predictor not found in student model. Adding new duration predictor...")
        try:
            from f5_tts.model import DurationPredictor
            student_duration_predictor = DurationPredictor(vocab_size, 512, 32, 3, 0.5)
            # Initialize on CPU first, will be moved to proper device later
            setattr(student_model, 'duration_predictor', student_duration_predictor)
            accelerator.print("Duration predictor successfully added to student model.")
        except ImportError:
            accelerator.print("[WARNING] Could not import DurationPredictor. Duration prediction will be disabled.")
        except Exception as e:
            accelerator.print(f"[WARNING] Error adding duration predictor: {e}. Duration prediction will be disabled.")
    else:
        accelerator.print("Duration predictor found in student model.")
        
    # --- Define Optimizer with Parameter Groups (Raw) ---
    optimizer_cls = AdamW
    if args.bnb_optimizer:
        try: import bitsandbytes as bnb; optimizer_cls = bnb.optim.AdamW8bit; accelerator.print("Using BNB 8-bit AdamW.")
        except ImportError: accelerator.print("[Warning] bitsandbytes not found. Falling back to AdamW.")
    
    # Create separate parameter groups for duration predictor and the rest of the model
    if args.use_duration_predictor and hasattr(student_model, 'duration_predictor') and student_model.duration_predictor is not None:
        # Get duration predictor parameters
        duration_params = list(student_model.duration_predictor.parameters())
        
        # Get all other parameters from the model
        other_params = [p for p in student_model.parameters() 
                       if not any(p is dp for dp in duration_params)]
        
        accelerator.print(f"Duration predictor params: {len(duration_params)}, Other params: {len(other_params)}")
        
        # Create parameter groups with different hyperparameters
        param_groups = [
            {'params': duration_params, 'lr': args.learning_rate * 3, 'weight_decay': 0.0003},
            {'params': other_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay}
        ]
        optimizer = optimizer_cls(
            param_groups, 
            betas=(0.9, 0.98),
            eps=1e-8
        )
        accelerator.print(f"Using optimizer with parameter groups - Duration predictor LR: {args.learning_rate * 3}, Others LR: {args.learning_rate}")
    else:
        # If no duration predictor, use all parameters with same settings
        optimizer = optimizer_cls(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-8)
        if optimizer_cls is AdamW: accelerator.print("Using standard AdamW.")
            
    # --- Define Dataset and Calculate Scheduler Steps ---
    accelerator.print(f"Loading dataset '{args.dataset_name}'...")
    try: train_dataset = load_dataset(args.dataset_name, args.tokenizer, mel_spec_kwargs=mel_spec_kwargs); accelerator.print(f"Dataset loaded. Size: {len(train_dataset)}")
    except Exception as e: accelerator.print(f"ERROR loading dataset: {e}"); exit(1)

    # --- Define Dataloader (Raw - needed for length calculation) ---
    # Note: This dataloader is temporary for length calculation if using frame batching.
    # The final dataloader used for training will be created *after* loading the checkpoint state.
    actual_batches_per_epoch_per_gpu = 0
    if args.batch_size_type == "sample":
         temp_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_per_gpu)
         actual_batches_per_epoch_per_gpu = len(temp_dataloader) # Batches per GPU per epoch
         del temp_dataloader
    elif args.batch_size_type == "frame":
         temp_sampler = SequentialSampler(train_dataset)
         temp_batch_sampler = DynamicBatchSampler(temp_sampler, frames_threshold=args.batch_size_per_gpu, max_samples=args.max_samples, random_seed=SEED, drop_residual=False)
         temp_dataloader = DataLoader(train_dataset, batch_sampler=temp_batch_sampler)
         actual_batches_per_epoch_per_gpu = len(temp_dataloader) # Batches per GPU per epoch
         del temp_dataloader, temp_batch_sampler, temp_sampler
    else: accelerator.print(f"ERROR: Invalid batch_size_type: {args.batch_size_type}"); exit(1)

    if actual_batches_per_epoch_per_gpu == 0 and len(train_dataset) > 0:
        accelerator.print("[Warning] Calculated 0 batches per epoch per GPU. Check batch_size_per_gpu / frame threshold. Training might not proceed.")

    total_updates = math.ceil(actual_batches_per_epoch_per_gpu / args.grad_accumulation_steps) * args.epochs
    accelerator.print(f"Scheduler Calculation: Batches/Epoch/GPU={actual_batches_per_epoch_per_gpu}, Total Updates={total_updates}")


    # --- Define Scheduler (Raw) ---
    warmup_updates = args.num_warmup_updates
    decay_updates = max(0, total_updates - warmup_updates)
    accelerator.print(f"Scheduler Setup: Total Updates={total_updates}, Warmup Updates={warmup_updates}, Decay Updates={decay_updates}")
    if warmup_updates >= total_updates and total_updates > 0: accelerator.print(f"[Warning] Warmup updates >= Total updates."); decay_updates = 0
    elif total_updates == 0: accelerator.print("[Warning] Total updates is 0. Setting warmup/decay to 0."); warmup_updates = 0; decay_updates = 0

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=max(0, warmup_updates))
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, decay_updates), eta_min=1e-8) # Ensure T_max >= 1
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates])

    # --- Initialize EMA (Raw) ---
    ema_model = None
    if args.use_ema:
        ema_model = EMA(student_model, beta=args.ema_decay, update_every=10) # Use raw student model
        accelerator.print(f"EMA initialized with decay {args.ema_decay}")

    # --- Load Checkpoint State (Optimizer, Scheduler, EMA) into RAW objects ---
    start_update, start_epoch = load_distill_checkpoint(
        accelerator, student_model, optimizer, scheduler, ema_model, args.output_dir
    )
    global_update = start_update
    
    # --- Calculate Resume Point ---
    skip_batches_in_epoch = 0  # Remove the start_epoch = 0 part
    
    # If user specified a resume epoch, use that directly
    if args.resume_epoch is not None:
        start_epoch = args.resume_epoch
        accelerator.print(f"Using manually specified resume epoch: {start_epoch}")
    elif start_update > 0:
        # Only calculate if not returned from checkpoint
        if start_epoch == 0:  # Only recalculate if the checkpoint didn't have epoch info
            updates_per_epoch = 6568  # Based on your observed values ONCE only
            start_epoch = start_update // updates_per_epoch
            accelerator.print(f"Calculated resume epoch: {start_epoch} (based on update {start_update})")
            
    # --- Create Final Dataloader (Raw) ---
    if args.batch_size_type == "sample":
         train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False, prefetch_factor=2 if args.num_workers > 0 else None, batch_size=args.batch_size_per_gpu, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    elif args.batch_size_type == "frame":
         sampler = SequentialSampler(train_dataset)
         batch_sampler = DynamicBatchSampler(sampler, frames_threshold=args.batch_size_per_gpu, max_samples=args.max_samples, random_seed=SEED, drop_residual=False)
         train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False, prefetch_factor=2 if args.num_workers > 0 else None, batch_sampler=batch_sampler)
         # Recalculate actual_batches_per_epoch_per_gpu with final dataloader if needed for sanity check or logging
         # actual_batches_per_epoch_per_gpu = len(train_dataloader)

    # --- Prepare objects with Accelerator (AFTER loading state) ---
    accelerator.print("Preparing model, optimizer, dataloader, scheduler with Accelerator...")
    student_model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        student_model, optimizer, train_dataloader, scheduler
    )
    accelerator.print("Preparation complete.")

    # --- Move EMA model to device (AFTER prepare) ---
    if ema_model is not None:
        ema_model.to(accelerator.device)
        accelerator.print(f"EMA model moved to device: {accelerator.device}")

    # --- Initialize Logger ---
    run_name = f"distill_{args.student_exp_name}_{args.dataset_name}_lr{args.learning_rate}_bs{args.batch_size_per_gpu}"
    tb_writer = None; wandb_run = None

    if accelerator.is_main_process:
        logger_to_use = args.logger
        if logger_to_use == "wandb":
            try: import wandb; wandb_run = wandb.init(project=f"F5TTS_Distill_{args.dataset_name}", name=run_name, config=vars(args), resume="allow", id=wandb.util.generate_id() if start_update == 0 else None); accelerator.print("Initialized WandB.")
            except ImportError: accelerator.print("[Warning] WandB not installed. Disabling WandB."); logger_to_use = None
            except Exception as e: accelerator.print(f"[Warning] WandB init failed: {e}. Disabling WandB."); logger_to_use = None
        elif logger_to_use == "tensorboard":
            try: tb_log_dir = os.path.join(args.output_dir, args.logging_dir, run_name); os.makedirs(tb_log_dir, exist_ok=True); tb_writer = SummaryWriter(log_dir=tb_log_dir); accelerator.print(f"Initialized TensorBoard: {tb_log_dir}")
            except Exception as e: accelerator.print(f"[Warning] TensorBoard init failed: {e}. Disabling TensorBoard."); logger_to_use = None
        args.logger = logger_to_use # Update args based on successful init

    # --- Initialize Vocoder ---
    vocoder = None; log_samples_path = None
    cfg_strength_sample = default_cfg_strength; nfe_step_sample = default_nfe_step; sway_sampling_coef_sample = default_sway_coef
    if args.log_samples and accelerator.is_main_process:
        if load_vocoder is None: accelerator.print("[Warning] Cannot log samples: 'load_vocoder' failed import."); args.log_samples = False
        else:
            try: accelerator.print(f"Loading vocoder '{mel_spec_type}'..."); vocoder = load_vocoder(vocoder_name=mel_spec_type); vocoder.to(accelerator.device); log_samples_path = os.path.join(args.output_dir, "samples"); os.makedirs(log_samples_path, exist_ok=True); accelerator.print(f"Sample logging enabled: {log_samples_path}")
            except Exception as e: accelerator.print(f"[Warning] Failed to load vocoder: {e}. Disabling sample logging."); args.log_samples = False

    # Sample setup
    if args.log_samples and accelerator.is_main_process and ref_audio_paths:
        ref_mels = preload_reference_mels(accelerator, ref_audio_paths, mel_spec_module)
        accelerator.print(f"Loaded {len(ref_mels)} reference mel spectrograms for consistent sampling")
        
    # --- Distillation Training Loop ---
    accelerator.print("Starting distillation training...")
    teacher_model.eval() # Already frozen and on device

    # --- Calculate Resume Point ---
    start_epoch = 0; skip_batches_in_epoch = 0
    if start_update > 0:
        # Force epoch calculation based directly on the update number
        # For example, if you're at update 178000, and you know there are around 6568 updates per epoch:
        start_epoch = (start_update // 6568)  # Direct division by known updates per epoch
        accelerator.print(f"DIRECT CALCULATION: start_update={start_update}, forced_epoch={start_epoch+1}")
        
        # Rest of your existing epoch calculation code
        if actual_batches_per_epoch_per_gpu > 0:
            updates_per_epoch_per_gpu = math.ceil(actual_batches_per_epoch_per_gpu / args.grad_accumulation_steps)
            if updates_per_epoch_per_gpu > 0:
                # Your original calculation - keep it for comparison
                calculated_epoch = (start_update - 1) // updates_per_epoch_per_gpu
                skip_updates_in_epoch = (start_update - 1) % updates_per_epoch_per_gpu
                skip_batches_in_epoch = skip_updates_in_epoch * args.grad_accumulation_steps
                accelerator.print(f"ORIGINAL CALC: start_update={start_update}, calculated_epoch={calculated_epoch+1}")
                # Override with the forced calculation
                accelerator.print(f"OVERRIDING epoch from {calculated_epoch+1} to {start_epoch+1}")
                
    # --- Epoch Loop ---
    for epoch in range(start_epoch, args.epochs):
        student_model.train() # Set prepared model to train mode

        # Set epoch for sampler
        if args.batch_size_type == "frame":
             if hasattr(train_dataloader, 'batch_sampler') and hasattr(train_dataloader.batch_sampler, 'set_epoch'): train_dataloader.batch_sampler.set_epoch(epoch + SEED)
             elif hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'): train_dataloader.sampler.set_epoch(epoch + SEED)
        elif args.batch_size_type == "sample" and hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, "set_epoch"):
             train_dataloader.sampler.set_epoch(epoch + SEED)

        # Skip batches if resuming within an epoch
        effective_dataloader = train_dataloader
        progress_bar_initial = 0
        is_resuming_epoch = (epoch == start_epoch and skip_batches_in_epoch > 0)
        if is_resuming_epoch:
             accelerator.print(f"Skipping first {skip_batches_in_epoch} batches for epoch {epoch + 1}...")
             effective_dataloader = accelerator.skip_first_batches(train_dataloader, skip_batches_in_epoch)
             progress_bar_initial = math.ceil(skip_batches_in_epoch / args.grad_accumulation_steps)

        # Progress bar setup
        total_batches_this_epoch_per_gpu = len(train_dataloader) # Use full length for display
        total_updates_this_epoch_per_gpu = math.ceil(total_batches_this_epoch_per_gpu / args.grad_accumulation_steps)
        progress_bar = tqdm(total=total_updates_this_epoch_per_gpu, initial=progress_bar_initial, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="update", disable=not accelerator.is_local_main_process, dynamic_ncols=True)

        batch_iterator = iter(effective_dataloader)
        processed_batches_in_epoch = skip_batches_in_epoch if is_resuming_epoch else 0

        # --- Batch Loop ---
        while processed_batches_in_epoch < total_batches_this_epoch_per_gpu:
            try: batch = next(batch_iterator)
            except StopIteration: accelerator.print("[Warning] Batch iterator stopped."); break

            # --- Gradient Accumulation Context ---
            with accelerator.accumulate(student_model): # Use prepared model

                # --- Prepare Inputs ---
                try:
                    mel_input = batch["mel"].to(accelerator.device).permute(0, 2, 1) # b d n -> b n d
                    text_list_input = batch["text"]
                    mel_lengths = batch["mel_lengths"].to(accelerator.device)
                    unwrapped_student = accelerator.unwrap_model(student_model)
                    if not hasattr(unwrapped_student, 'vocab_char_map') or not unwrapped_student.vocab_char_map: raise ValueError("Student vocab_char_map missing.")
                    text_tensor_input = list_str_to_idx(text_list_input, unwrapped_student.vocab_char_map, padding_value=-1).to(accelerator.device)
                except Exception as e: accelerator.print(f"ERROR preparing batch data: {e}"); processed_batches_in_epoch +=1; continue

                # --- CFM Forward Logic ---
                batch_size, seq_len, _ = mel_input.shape; dtype = mel_input.dtype; device = accelerator.device
                mask = lens_to_mask(mel_lengths, length=seq_len).to(device)
                frac_lengths = torch.zeros((batch_size,), device=device).float().uniform_(0.7, 1.0)
                rand_span_mask = mask_from_frac_lengths(mel_lengths, frac_lengths).to(device)
                if exists(mask): rand_span_mask &= mask
                x1 = mel_input; x0 = torch.randn_like(x1, device=device); time = torch.rand((batch_size,), dtype=dtype, device=device)
                t_unsqueezed = time.unsqueeze(-1).unsqueeze(-1); xt = (1 - t_unsqueezed) * x0 + t_unsqueezed * x1
                target_flow = x1 - x0; cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1, device=device), x1)

                # --- Teacher Forward ---
                with torch.no_grad():
                     teacher_model.eval()
                     try: teacher_pred_flow = teacher_model.transformer(x=xt, cond=cond, text=text_tensor_input, time=time, drop_audio_cond=False, drop_text=False)
                     except Exception as e: accelerator.print(f"ERROR Teacher forward: {e}"); processed_batches_in_epoch +=1; continue

                # --- Student Forward ---
                student_model.train() # Ensure prepared model is in train mode
                unwrapped_student_model = accelerator.unwrap_model(student_model)
                drop_audio_cond = random.random() < unwrapped_student_model.audio_drop_prob
                drop_text = (random.random() < unwrapped_student_model.cond_drop_prob)
                if drop_text: drop_audio_cond = True
                try: student_pred_flow = unwrapped_student_model.transformer(x=xt, cond=cond, text=text_tensor_input, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text)
                except Exception as e: accelerator.print(f"ERROR Student forward: {e}"); processed_batches_in_epoch +=1; continue

                # --- Calculate Losses ---
                try:
                    student_loss_full = F.mse_loss(student_pred_flow, target_flow, reduction="none")
                    masked_student_loss = student_loss_full * rand_span_mask.unsqueeze(-1)
                    student_loss = masked_student_loss.sum() / rand_span_mask.sum().clamp(min=1)
                    
                    if args.distill_loss_type == "mse": 
                        distill_loss_full = F.mse_loss(student_pred_flow, teacher_pred_flow.detach(), reduction="none")
                    elif args.distill_loss_type == "l1": 
                        distill_loss_full = F.l1_loss(student_pred_flow, teacher_pred_flow.detach(), reduction="none")
                    else: 
                        raise ValueError(f"Unsupported distill_loss_type: {args.distill_loss_type}")

                    if args.spec_l1_weight > 0.0:
                        if args.spec_l1_weight > 0.0:
                            spec_l1_full = F.l1_loss(student_pred_flow,
                                                     teacher_pred_flow.detach(),
                                                     reduction="none")
                            masked_spec = spec_l1_full * rand_span_mask.unsqueeze(-1)
                            spec_l1 = masked_spec.sum() / rand_span_mask.sum().clamp(min=1)
                        else:
                            # zero scalar tensor on the correct device
                            spec_l1 = torch.tensor(0.0, device=student_pred_flow.device)
                        
                    masked_distill_loss = distill_loss_full * rand_span_mask.unsqueeze(-1)
                    distill_loss = masked_distill_loss.sum() / rand_span_mask.sum().clamp(min=1)
                    alpha = args.distill_loss_weight; 

                    # Calc loss with duration predictor loss
                    total_loss = (1.0 - alpha) * student_loss + alpha * distill_loss + spec_l1 * args.spec_l1_weight
                    
                    # In the loss calculation section:
                    if args.use_duration_predictor and hasattr(unwrapped_student_model, 'duration_predictor') and unwrapped_student_model.duration_predictor is not None and 'attn' in batch:
                        text_tokens = text_tensor_input
                        b, nt = text_tokens.shape
                        text_lengths = batch.get("text_lengths", torch.full((b,), nt, device=device))
                        attn = batch['attn']
                        
                        # Create mask for text tokens
                        range_tensor = torch.arange(nt, device=device).unsqueeze(0)
                        text_tokens_mask = (range_tensor < text_lengths.unsqueeze(1)).int()
                        
                        # Calculate durations from attention
                        w = attn.sum(dim=2)
                        logw_ = torch.log(w + 1e-6) * text_tokens_mask
                        
                        # Get duration predictions
                        logw = unwrapped_student_model.duration_predictor(text_tokens, text_tokens_mask)
                        
                        # Calculate duration loss
                        l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(text_tokens_mask)
                        dur_loss = torch.sum(l_length.float())
                        
                        # Add duration loss with weight
                        duration_loss_weight = getattr(args, 'duration_loss_weight', 0.5)
                        total_loss = total_loss + duration_loss_weight * dur_loss
                        
                        # Log duration loss
                        log_data["loss/duration"] = dur_loss.item()     
                        
                    if torch.isnan(total_loss) or torch.isinf(total_loss): accelerator.print(f"[ERROR] NaN/Inf loss at update {global_update}! Skipping."); optimizer.zero_grad(); processed_batches_in_epoch +=1; continue
                except Exception as e: accelerator.print(f"ERROR Loss calculation: {e}"); processed_batches_in_epoch +=1; continue

                # --- Backward Pass ---
                accelerator.backward(total_loss)

                # --- Steps and Logging (Executes only when gradients are synced) ---
                if accelerator.sync_gradients:
                    grad_norm = None
                    if args.max_grad_norm > 0: grad_norm = accelerator.clip_grad_norm_(student_model.parameters(), args.max_grad_norm) # Clip prepared model
                    optimizer.step()    # Step prepared optimizer
                    scheduler.step()    # Step prepared scheduler
                    optimizer.zero_grad()

                    if ema_model is not None and accelerator.is_main_process: ema_model.update() # Update raw EMA object

                    # --- Logging ---
                    if accelerator.is_main_process:
                        lr = scheduler.get_last_lr()[0]
                        log_data = {"loss/total": total_loss.item(), "loss/student": student_loss.item(), "loss/distill": distill_loss.item(), "learning_rate": lr}
                        if grad_norm is not None: log_data["gradient_norm"] = grad_norm.item()

                        if args.logger == "wandb" and wandb_run:
                             try: wandb.log(log_data, step=global_update)
                             except Exception as e: accelerator.print(f"[Warning] WandB log failed: {e}")
                        elif args.logger == "tensorboard" and tb_writer:
                            try:
                                for key, value in log_data.items(): tb_writer.add_scalar(key, value, global_step=global_update)
                            except Exception as e: accelerator.print(f"[Warning] TensorBoard log failed: {e}")

                        postfix_data = {"loss": f"{total_loss.item():.4f}", "lr": f"{lr:.2e}"}
                        if grad_norm is not None: postfix_data["grad"] = f"{grad_norm.item():.2f}"
                        progress_bar.set_postfix(**postfix_data)

                    # --- Checkpointing ---
                    if accelerator.is_main_process:
                        if global_update > 0 and global_update % args.save_per_updates == 0:
                            # Pass prepared objects; EMA is raw but handled inside save_checkpoint
                            save_distill_checkpoint(accelerator, student_model, optimizer, scheduler, ema_model, global_update, epoch, args.output_dir, args.keep_last_n_checkpoints, is_last=False)

                            # --- Sample Logging ---
                            if args.log_samples and vocoder is not None:
                                try:
                                    accelerator.print(f"Generating samples for update {global_update}...")
                                    model_for_sampling = ema_model.ema_model if args.use_ema and ema_model is not None else accelerator.unwrap_model(student_model)
                                    model_for_sampling.eval()
                                    ref_audio_len = mel_lengths[0].item()
                                    first_text = text_list_input[0]; first_text_str = "".join(first_text) if isinstance(first_text, list) else str(first_text)
                                    infer_text = [first_text_str + " " + first_text_str]
                                    target_sr = model_for_sampling.mel_spec.target_sample_rate

                                    with torch.inference_mode():
                                        generated_mel, _ = model_for_sampling.sample(cond=mel_input[0][:ref_audio_len].unsqueeze(0), text=infer_text, duration=ref_audio_len * 2, steps=nfe_step_sample, cfg_strength=cfg_strength_sample, sway_sampling_coef=sway_sampling_coef_sample)
                                        generated_mel = generated_mel.to(torch.float32)
                                        gen_mel_only = generated_mel[:, ref_audio_len:, :] if generated_mel.shape[1] > ref_audio_len else generated_mel
                                        gen_mel_vocoder = gen_mel_only.permute(0, 2, 1).to(accelerator.device)
                                        ref_mel_vocoder = batch["mel"][0].unsqueeze(0).to(accelerator.device)
                                        if mel_spec_type == "vocos": gen_audio, ref_audio = vocoder.decode(gen_mel_vocoder).cpu(), vocoder.decode(ref_mel_vocoder).cpu()
                                        else: gen_audio, ref_audio = torch.zeros((1,1)), torch.zeros((1,1)) # Placeholder

                                    save_path_gen = os.path.join(log_samples_path, f"update_{global_update}_gen.wav")
                                    save_path_ref = os.path.join(log_samples_path, f"update_{global_update}_ref.wav")
                                    torchaudio.save(save_path_gen, gen_audio, target_sr); torchaudio.save(save_path_ref, ref_audio, target_sr)
                                    accelerator.print(f"Saved sample audio to {log_samples_path}")

                                    # Generate fixed reference samples if available
                                    if ref_mels:
                                        try:
                                            generate_reference_samples(
                                                accelerator, student_model, ema_model, global_update, vocoder, 
                                                log_samples_path, ref_texts, ref_mels, ref_sample_text_prompts,
                                                nfe_step_sample, cfg_strength_sample, sway_sampling_coef_sample
                                            )
                                        except Exception as e:
                                            accelerator.print(f"Warning: Error generating reference samples: {e}")
                                            import traceback
                                            traceback.print_exc()
                                    
                                    student_model.train() # Set back to train
                                except Exception as sample_err: accelerator.print(f"[Warning] Sample generation failed: {sample_err}"); student_model.train()

                        # Frequent "last" checkpoint
                        if global_update > 0 and global_update % args.last_per_updates == 0:
                            save_distill_checkpoint(accelerator, student_model, optimizer, scheduler, ema_model, global_update, epoch, args.output_dir, args.keep_last_n_checkpoints, is_last=True)

                    # --- Increment Update Counter ---
                    global_update += 1
                    progress_bar.update(1)
            # --- End Accumulation Block ---

            processed_batches_in_epoch += 1
        # --- End Batch Loop ---

        accelerator.wait_for_everyone()
        progress_bar.close()

        # --- Save Checkpoint at End of Epoch ---
        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch + 1} finished. Saving epoch end checkpoint.")
            save_distill_checkpoint(accelerator, student_model, optimizer, scheduler, ema_model, global_update, epoch, args.output_dir, args.keep_last_n_checkpoints, is_last=False) # Save numbered

        if accelerator.device.type == 'cuda': torch.cuda.empty_cache()
        elif accelerator.device.type == 'xpu': torch.xpu.empty_cache()
        gc.collect()
    # --- End Epoch Loop ---

    # --- End of Training ---
    accelerator.wait_for_everyone()
    accelerator.print("Distillation training finished.")

    if accelerator.is_main_process:
        accelerator.print("Saving final 'last' checkpoint...")
        save_distill_checkpoint(accelerator, student_model, optimizer, scheduler, ema_model, global_update, epoch, args.output_dir, args.keep_last_n_checkpoints, is_last=True)
        # Close loggers
        if args.logger == "wandb" and wandb_run:
             try: wandb.finish(); accelerator.print("WandB logger finished.")
             except Exception as e: accelerator.print(f"[Warning] Error closing WandB: {e}")
        elif args.logger == "tensorboard" and tb_writer:
             try: tb_writer.close(); accelerator.print("TensorBoard logger closed.")
             except Exception as e: accelerator.print(f"[Warning] Error closing TensorBoard writer: {e}")

    accelerator.print("Script finished.")

if __name__ == "__main__":
    main()
# --- END OF FILE distil (10).py ---