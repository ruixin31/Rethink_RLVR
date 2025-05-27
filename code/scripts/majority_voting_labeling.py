import os
import json
from collections import Counter
from tqdm import tqdm
from vllm import LLM, SamplingParams
from ttrl.verifier.qwen.qwen_math_parser import extract_answer
from ttrl.verifier.qwen.math_grade import grade_answer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_rollout", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--output_traces", action="store_true")
    args = parser.parse_args()
    
    # load vllm model
    model = LLM(model=args.model_name)
    sampling_params = SamplingParams(
        max_tokens=3072,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
    )
    
    # load data
    with open(args.data_path, "r") as f:
        data = json.load(f)
    print(len(data))
    
    args.output_path = args.output_path.replace(".jsonl", f"_worker{args.worker_id}of{args.num_workers}.jsonl")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            existing_data = [json.loads(line) for line in f.readlines()]
        existing_ids = {ex["id"] for ex in existing_data}
        if len(existing_ids) < len(existing_data):
            print(f"Warning: Duplicate IDs found in {args.output_path}.")
            with open(args.output_path, "w") as f:
                for id in existing_ids:
                    ex = existing_data[id]
                    f.write(json.dumps(ex) + "\n")
    else:
        existing_ids = set()
    
    # label data
    for i, ex in tqdm(enumerate(data)):
        if i % args.num_workers != args.worker_id:
            continue
        if i in existing_ids:
            continue
        prompt = ex["prompt"]
        answer = ex["answer"]
        preds = []
        corrects = []
        traces = []
        
        outputs = model.generate([prompt]*args.num_rollout, sampling_params)
        for output in outputs:
            output = output.outputs[0].text
            traces.append(output)
            pred = extract_answer(output, "math")
            accuracy = 1.0 if grade_answer(pred, answer) else 0.0
            preds.append(pred)
            corrects.append(accuracy)
            
        counter = Counter(preds)
        majority_pred, _ = counter.most_common(1)[0]
        majority_correct = 1.0 if grade_answer(majority_pred, answer) else 0.0
        average_correct = sum(corrects) / len(corrects)
        
        ex_results = {
            "id": i,
            "prompt": prompt,
            "answer": answer,
            "preds": preds,
            "corrects": corrects,
            "average_correct": average_correct,
            "majority_pred": majority_pred,
            "majority_correct": majority_correct,
        }
        if args.output_traces:
            ex_results["outputs"] = traces
        print(preds)
        print(majority_pred)
        print(majority_correct)
        
        with open(args.output_path, "a+") as f:
            f.write(json.dumps(ex_results) + "\n")