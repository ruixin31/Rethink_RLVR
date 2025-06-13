import argparse
import json
import os
from typing import Optional

import torch
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ttrl.verifier.auto_verify import auto_verify

BENCHMARKS = {
    "MATH-500": {
        "path": "data/MATH-TTT/test.json",
        "rollouts": 1,
    },
    "AIME-2024": {
        "path": "data/AIME-TTT/test.json",
        "rollouts": 8,
    },
    "AIME-2025": {
        "path": "data/AIME2025-TTT/test.json",
        "rollouts": 8,
    },
    "AMC": {
        "path": "data/AMC-TTT/test.json",
        "rollouts": 8,
    },
}

# Default prompts
BASE_MODEL_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n<|user|>\n{}\n<|assistant|>\n<think>"


def get_chat_prompts(
    tokenizer: AutoTokenizer,
    prompt: str,
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate chat prompts using the tokenizer's chat template.

    Args:
        tokenizer: The tokenizer to use for formatting
        prompt: The user prompt
        system_prompt: Optional system prompt

    Returns:
        Formatted chat prompt string
    """
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def process(args: argparse.Namespace) -> None:
    """
    Process math problems through a language model and evaluate performance.

    Args:
        args: Command line arguments containing model path, datasets, and other parameters
    """
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = LLM(args.model_path)

    # Process each dataset
    for dataset_name in args.datasets.split(","):
        if dataset_name not in BENCHMARKS:
            print(f"Warning: Dataset {dataset_name} not found in BENCHMARKS. Skipping...")
            continue

        print(f"Evaluating {dataset_name}...")

        dataset_info = BENCHMARKS[dataset_name]
        test_path = dataset_info["path"]
        rollouts = dataset_info["rollouts"]

        # Create output filename
        model_name = args.model_path.split("/")[-1]
        file_name = f"{model_name}_{dataset_name}.json"

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Load test data
        with open(test_path) as f:
            test_data = json.load(f)

        # Extract labels and prepare indices
        all_labels = [x['answer'] for x in test_data]
        all_indices = list(range(len(test_data)))
        
        # Process prompts based on model type
        if args.is_base_model:
            formatted_prompts = [BASE_MODEL_PROMPT.format(x['prompt']) for x in test_data]
            stop_string = ["</answer>"]
            include_stop_str_in_output = True
        else:
            formatted_prompts = [get_chat_prompts(tokenizer, x['prompt']) for x in test_data]
            stop_string = None
            include_stop_str_in_output = False

        if dataset_info["rollouts"] == 1:
            temperature = args.temperature
        else:
            temperature = args.temperature_at_k

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=3072,
            include_stop_str_in_output=include_stop_str_in_output,
            stop=stop_string,
        )

        # Prepare prompts for batch processing
        all_prompts = sum([[prompt] * rollouts for prompt in formatted_prompts], [])
        all_outputs = []
        batch_size = (len(all_prompts) + args.shards - 1) // args.shards

        # Process in batches
        for i in range(args.shards):
            shard_prompts = all_prompts[i * batch_size: (i + 1) * batch_size]
            all_output_ref = model.generate(
                shard_prompts,
                sampling_params=sampling_params,
            )
            all_outputs.append(all_output_ref)

        # Reshape outputs
        all_outputs_list = sum(all_outputs, [])
        all_outputs = []
        for i, prompt in enumerate(formatted_prompts):
            all_outputs.append(all_outputs_list[i * rollouts: (i + 1) * rollouts])

        all_outputs = list(zip(*all_outputs))

        # Calculate accuracies
        all_accuracies = []
        verify_task = "math"
        for outputs in all_outputs:
            all_accuracies.append(auto_verify(verify_task, 1, outputs, all_labels))

        accuracy_at_k = np.mean([np.mean(acc) for acc in all_accuracies])
        temp_all_accuracies = np.array(all_accuracies)
        pass_at_k = temp_all_accuracies.max(axis=0).mean()

        # Prepare metadata
        metadata = []
        for prompt, label, index in zip(formatted_prompts, all_labels, all_indices):
            metadata.append({
                "prompt": prompt,
                "label": label,
                "index": index,
                "outputs": []
            })

        for outputs in all_outputs:
            for idx, output in enumerate(outputs):
                metadata[idx]["outputs"].append(output.outputs[0].text)

        # Save results
        output = {
            f"avg@{rollouts}": accuracy_at_k,
            f"pass@{rollouts}": pass_at_k,
            "dataset": dataset_name,
            "model": args.model_path,
            "temperature": temperature,
            "rollouts": rollouts,
            "include_stop_str_in_output": include_stop_str_in_output,
            "stop_string": stop_string,
            "gpu_type": torch.cuda.get_device_name(),
            "shards": args.shards,
            "metadata": metadata
        }

        output_file = os.path.join(args.output_dir, file_name)
        with open(output_file, "w") as f:
            json.dump(output, f, indent=4)
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--datasets", type=str, required=True,
                        help="List of dataset names to evaluate on")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for single rollout")
    parser.add_argument("--temperature_at_k", type=float, default=0.6,
                        help="Temperature for sampling at k")
    parser.add_argument("--is_base_model", action="store_true",
                        help="Whether the model is a base model and should apply the base model prompt")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_outputs",
                        help="Output directory")
    parser.add_argument("--shards", type=int, default=4,
                        help="Number of shards of eval suite to create for an evaluation for reproducibility purposes. Note both GPU and this number need to match to obtain original results for temperature 0 runs",
    )
    args = parser.parse_args()

    process(args)