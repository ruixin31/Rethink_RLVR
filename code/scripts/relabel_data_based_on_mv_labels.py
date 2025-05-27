import os
import random
import json
from collections import Counter
from ttrl.verifier.qwen.math_grade import grade_answer
from tqdm import tqdm


def get_mv_labels(data_path, filter='all', mv_order=0):
    data = []
    num_correct = 0
    num_incorrect = 0
    num_empty = 0
    
    if not os.path.isdir(data_path):
        data_paths = [data_path]
    else:
        data_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".jsonl")]
    
    for data_path in tqdm(data_paths):
        local_num_correct = 0
        local_num_incorrect = 0
        with open(data_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                id = ex["id"]
                preds = ex["preds"]
                corrects = ex["corrects"]
                if filter.startswith('code_'):
                    average_has_code = ex["average_has_code"]
                else:
                    average_has_code = None
                
                # Count predictions
                preds = [pred for pred in preds if pred]
                counter = Counter(preds)
                # Get predictions sorted by frequency
                sorted_preds = [pred for pred, _ in counter.most_common()]
                
                # Get the nth most voted prediction (0-based index)
                if mv_order < len(sorted_preds):
                    majority_pred = sorted_preds[mv_order]
                    # Check if this prediction is correct
                    majority_correct = 1.0 if grade_answer(majority_pred, ex["answer"]) else 0.0
                else:
                    # If mv_order is beyond available predictions, use the last one
                    if not sorted_preds:
                        majority_pred = ''
                        num_empty += 1
                    else:
                        majority_pred = sorted_preds[-1]
                    majority_correct = 1.0 if grade_answer(majority_pred, ex["answer"]) else 0.0
                
                local_num_correct += 1 if majority_correct else 0
                local_num_incorrect += 1 if not majority_correct else 0
                
                # Update the example with the selected prediction
                ex["majority_pred"] = majority_pred
                ex["majority_correct"] = majority_correct
                ex["sorted_preds"] = sorted_preds
                
                if filter == 'all':
                    data.append(ex)
                elif filter == 'correct':
                    if majority_correct:
                        data.append(ex)
                elif filter == 'incorrect':
                    if not majority_correct:
                        data.append(ex)
                elif filter == 'pass_at_64_not_correct':
                    if not any(corrects):
                        # breakpoint()
                        data.append(ex)
                elif filter.startswith('code_'):
                    code_percent = int(filter.split('_')[1])
                    print(average_has_code, code_percent / 100)
                    if average_has_code <= code_percent / 100:
                        data.append(ex)
                    
        
        num_correct += local_num_correct
        num_incorrect += local_num_incorrect
    
    print(f"Global stats for {filter} filtering:")
    print(f"num_correct: {num_correct}, num_incorrect: {num_incorrect}, num_empty: {num_empty}")
    print(f"length of data: {len(data)}")
    return data


def relabel_data(data_path, output_dir, model_id, filter='all', num_test_samples=100, mv_order=0, use_gt_answer=False, use_minority_pred=False, use_random_answer=False):
    if filter == 'all':
        source = f"deepscaler_mv_labeled_{model_id}"
        output_path = os.path.join(output_dir, f"DeepScaleR_mv_labeled_{model_id}/train.json")
    elif filter == 'correct':
        source = f"deepscaler_mv_labeled_{model_id}_correct"
        output_path = os.path.join(output_dir, f"DeepScaleR_mv_labeled_{model_id}_correct/train.json")
    elif filter == 'incorrect':
        source = f"deepscaler_mv_labeled_{model_id}_incorrect"
        output_path = os.path.join(output_dir, f"DeepScaleR_mv_labeled_{model_id}_incorrect/train.json")
    elif filter == 'pass_at_64_not_correct':
        source = f"deepscaler_mv_labeled_{model_id}_pass_at_64_not_correct"
        output_path = os.path.join(output_dir, f"DeepScaleR_gt_answer{model_id}_pass_at_64_not_correct/train.json")
    elif filter.startswith('code_'):
        code_percent = int(filter.split('_')[1])
        source = f"deepscaler_mv_labeled_{model_id}_code_{code_percent}"
        output_path = os.path.join(output_dir, f"DeepScaleR_gt_answer{model_id}_code_{code_percent}/train.json")
    if mv_order != 0:
        output_path = output_path.replace("/train.json", f"_mv_order_{mv_order}/train.json")
    if use_gt_answer:
        output_path = output_path.replace("mv_labeled", "mv_labeled_but_gt_answer")
    if use_minority_pred:
        output_path = output_path.replace("mv_labeled", "mv_labeled_but_outlier_pred")
    if use_random_answer:
        output_path = output_path.replace("mv_labeled", "mv_labeled_but_random_answer")
    assert not use_minority_pred or not use_gt_answer

    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = get_mv_labels(data_path, filter=filter, mv_order=mv_order)

    formatted_data = []

    if use_random_answer:
        all_possible_answers = [ex['answer'] for ex in data]
        random.seed(42)
        random.shuffle(all_possible_answers)
        for ex, answer in zip(data, all_possible_answers):
            ex['random_answer'] = answer



    for ex in data:
        answer = ex["majority_pred"] if not use_gt_answer else ex["answer"]
        if use_random_answer:
            answer = ex["random_answer"]
        if use_minority_pred:
            answer = ex["sorted_preds"][-1]
        formatted_data.append({
            "id": ex["id"],
            "prompt": ex["prompt"],
            "answer": answer,
            "source": source,
        })
    
    test_data = formatted_data[:num_test_samples]
    train_data = formatted_data[num_test_samples:]
    
    with open(output_path, "w") as f:
        json.dump(train_data, f, indent=4)

    test_output_path = output_path.replace("train.json", "test.json")
    with open(test_output_path, "w") as f:
        json.dump(test_data, f, indent=4)


if __name__ == "__main__":
    data_path = "code/data/mv_labeled_qwen2.5_math_7b/merged"
    model_id = "qwen2.5_math_7b"
    output_dir = "code/data/"

    relabel_data(data_path, output_dir, model_id, mv_order=0)
