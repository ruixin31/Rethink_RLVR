import argparse
import shutil
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Export a DeepSpeed checkpoint to HuggingFace format')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the DeepSpeed checkpoint directory')
    parser.add_argument('--step', type=int, required=True,
                      help='Checkpoint step number to export')
    parser.add_argument('--base-model', type=str, default="Qwen/Qwen2.5-Math-7B",
                      help='Base model name (default: Qwen/Qwen2.5-Math-7B)')
    parser.add_argument('--output-dir', type=str, default="./export-for-eval",
                      help='Directory to save the exported model (default: ./export-for-eval)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load base model configuration
    print(f"Loading base model configuration from {args.base_model}")
    cfg = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_config(cfg)
    
    # 2. Load checkpoint state dict
    print(f"Loading checkpoint from {args.checkpoint} at step {args.step}")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        args.checkpoint, 
        tag=f"global_step{args.step}"
    )
    
    # 3. Load state dict into model
    print("Loading state dict into model")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 4. Load tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # 5. Save model and tokenizer
    print(f"Saving model and tokenizer to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
if __name__ == "__main__":
    main() 