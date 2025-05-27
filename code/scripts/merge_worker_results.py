import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from ttrl.verifier.qwen.qwen_eval import qwen_reward_fn, test_time_train, format_reward_fn, dense_reward_fn, box_only_format_reward_fn, inverse_box_only_format_reward_fn, inverse_qwen_reward_fn, contain_python_reward_fn, contain_python_wo_backticks_reward_fn

def merge_worker_results(input_pattern, output_path):
    # Get all worker files matching the pattern
    base_dir = os.path.dirname(input_pattern)
    base_name = os.path.basename(input_pattern)
    worker_files = []
    
    for file in os.listdir(base_dir):
        if file.startswith(base_name.replace(".jsonl", "")) and file.endswith(".jsonl"):
            worker_files.append(os.path.join(base_dir, file))
    
    if not worker_files:
        raise ValueError(f"No worker files found matching pattern: {input_pattern}")
    
    print(f"Found {len(worker_files)} worker files")
    
    # Aggregate results by ID
    results_by_id = defaultdict(lambda: {
        "id": None,
        "prompt": None,
        "answer": None,
        "preds": [],
        "corrects": [],
        "average_correct": 0.0,
        "majority_pred": None,
        "majority_correct": 0.0
    })
    
    # Read all results
    for worker_file in worker_files:
        with open(worker_file, 'r') as f:
            for line in f:
                result = json.loads(line.strip())
                ex_id = result['id']
                result['has_code'] = list(map(contain_python_wo_backticks_reward_fn, result["outputs"]))
                
                # Initialize if first time seeing this ID
                if results_by_id[ex_id]["id"] is None:
                    results_by_id[ex_id].update({
                        "id": ex_id,
                        "prompt": result["prompt"],
                        "answer": result["answer"],
                        "has_code": result["has_code"]
                    })
                
                # Aggregate predictions and correctness
                results_by_id[ex_id]["preds"].extend(result["preds"])
                results_by_id[ex_id]["corrects"].extend(result["corrects"])
                results_by_id[ex_id]["has_code"].extend(result["has_code"])
    
    # Calculate final statistics for each example
    all_results = []
    for ex_id, result in sorted(results_by_id.items()):
        # Calculate new average correctness
        result["average_correct"] = sum(result["corrects"]) / len(result["corrects"])

        # Calculate new average has code
        result["average_has_code"] = sum(result["has_code"]) / len(result["has_code"])
        
        # Calculate new majority prediction
        from collections import Counter
        counter = Counter(result["preds"])
        result["majority_pred"] = counter.most_common(1)[0][0]
        
        # Calculate new majority correctness
        from ttrl.verifier.qwen.math_grade import grade_answer
        result["majority_correct"] = 1.0 if grade_answer(result["majority_pred"], result["answer"]) else 0.0

        
        all_results.append(result)
    
    # Write merged results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for result in tqdm(all_results, desc="Writing merged results"):
            f.write(json.dumps(result) + '\n')
    
    print(f"Merged {len(all_results)} results into {output_path}")
    print(f"Total predictions per example: {len(all_results[0]['preds']) if all_results else 0}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pattern", type=str, required=True,
                      help="Pattern for input files (e.g., 'path/to/results_worker*.jsonl')")
    parser.add_argument("--output_path", type=str, required=True,
                      help="Path for the merged output file")
    args = parser.parse_args()
    
    merge_worker_results(args.input_pattern, args.output_path) 