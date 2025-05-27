from tqdm import tqdm
from ttrl.verifier.qwen.qwen_eval import qwen_reward_fn, test_time_train, format_reward_fn, dense_reward_fn, box_only_format_reward_fn, inverse_box_only_format_reward_fn, inverse_qwen_reward_fn, contain_python_reward_fn, contain_python_wo_backticks_reward_fn
from ttrl.verifier.repetition_detection import no_repetition_reward_fn
import numpy as np
import random

def random_reward_fn(generated_text, golden_answer, task="math", is_r1_style=False, rate=0.5):
    return int(random.random() < rate)

def auto_verify(task, batch_size, all_outputs, all_labels):
    if "_AND_" in task:
        tasks = task.split("_AND_")
        rewards = [auto_verify(task, batch_size, all_outputs, all_labels) for task in tasks]
        # rewards: [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        return [1.0 if all([rewards[task_idx][batch_idx] for task_idx in range(len(tasks))]) else 0.0 for batch_idx in range(len(rewards[0]))]

    all_outputs = [output.outputs[0].text for output in all_outputs]

    task2verify = {
        "math": qwen_reward_fn,
        "inverse_math": inverse_qwen_reward_fn,
        "ttt": test_time_train,
        "format": format_reward_fn,
        "box_only_format": box_only_format_reward_fn,
        "inverse_box_only_format": inverse_box_only_format_reward_fn,
        # "math_r1_style": qwen_reward_fn,
        # "box_only_format_r1_style": box_only_format_reward_fn,
        # "inverse_math_r1_style": inverse_qwen_reward_fn,
        "dense": dense_reward_fn,
        "random": random_reward_fn,
        "contain_python": contain_python_reward_fn,
        "contain_python_wo_backticks": contain_python_wo_backticks_reward_fn,
        "no_repetition": no_repetition_reward_fn,
    }

    inverse_task = False
    if task.startswith("inverse_"):
        task = task.replace("inverse_", "")
        inverse_task = True

    is_r1_style = False
    if task.endswith("_r1_style"):
        task = task.replace("_r1_style", "")
        is_r1_style = True

    if task.startswith("random"):
        rate = float(task.replace("random", ""))
        task = "random"

    assert task in task2verify, f"{task} not in {list(task2verify.keys())}"

    verify_fn = task2verify[task]
    if task in ["math", "inverse_math"]:
        rewards = [verify_fn(output, label, is_r1_style=is_r1_style)
                   for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]
    elif task in ["ttt"]:
        rewards = []
        n_prompts = len(all_outputs) // batch_size
        for prompt_idx in range(n_prompts):
            group_outputs = all_outputs[batch_size *
                                        prompt_idx:batch_size*(prompt_idx+1)]
            group_labels = all_labels[batch_size *
                                      prompt_idx:batch_size*(prompt_idx+1)]
            rewards.extend(verify_fn(group_outputs, group_labels))
    elif task in ["format", "box_only_format", "inverse_box_only_format"]:
        rewards = [verify_fn(output, label, is_r1_style=is_r1_style)
                   for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]
    # elif task in ["box_only_format_r1_style", "inverse_box_only_format_r1_style"]:
    #     rewards = [verify_fn(output, label, is_r1_style=is_r1_style)
    #                for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]
    # elif task in ["math_r1_style", "inverse_math_r1_style"]:
    #     rewards = [verify_fn(output, label, is_r1_style=is_r1_style)
    #                for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]
    elif task in ["dense"]:
        rewards = [verify_fn(output, label, is_r1_style=is_r1_style)
                   for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]
    elif task in ["random"]:
        rewards = [verify_fn(output, label, rate=rate, is_r1_style=is_r1_style)
                   for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]
    elif task in ["contain_python"]:
        rewards = [verify_fn(output, label, is_r1_style=is_r1_style)
                   for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]
    elif task in ["contain_python_wo_backticks"]:
        rewards = [verify_fn(output, label, is_r1_style=is_r1_style)
                   for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]
    elif task in ["no_repetition"]:
        rewards = [verify_fn(output, label, is_r1_style=is_r1_style)
                   for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]

    if inverse_task:
        rewards = [1.0 - reward for reward in rewards]

    return rewards
