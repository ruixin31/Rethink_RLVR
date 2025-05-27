import concurrent.futures
from typing import List
import json
from collections import Counter
from ttrl.verifier.qwen.qwen_math_parser import extract_answer, extract_answer_box_only
from ttrl.verifier.qwen.math_grade import grade_answer
from ttrl.verifier.qwen.grader import math_equal as qwen_math_equal

from ttrl.verifier.qwen.grader import math_equal

def qwen_reward_fn(generated_text, golden_answer, task="math", is_r1_style=False):
    if golden_answer in ["A", "B", "C", "D"]:
        task = "gpqa"

    model_answer = extract_answer(generated_text, task, is_r1_style=is_r1_style)
    accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    return accuracy

def inverse_qwen_reward_fn(generated_text, golden_answer, task="math", is_r1_style=False):
    return 1.0 - qwen_reward_fn(generated_text, golden_answer, task, is_r1_style)

def format_reward_fn(generated_text, golden_answer, task="math", is_r1_style=False):
    model_answer = extract_answer(generated_text, task, is_r1_style=is_r1_style)
    return 1.0 if model_answer else 0.0

def box_only_format_reward_fn(generated_text, golden_answer, task="math", is_r1_style=False):
    model_answer = extract_answer_box_only(generated_text, task, is_r1_style=is_r1_style)
    return 1.0 if model_answer else 0.0

def inverse_box_only_format_reward_fn(generated_text, golden_answer, task="math", is_r1_style=False):
    return 1.0 - box_only_format_reward_fn(generated_text, golden_answer, task, is_r1_style)

def dense_reward_fn(generated_text, golden_answer, task="math", is_r1_style=False):
    golden_answer = json.loads(golden_answer)
    for label, label_score in golden_answer.items():
        model_answer = extract_answer(generated_text, task, is_r1_style)
        if grade_answer(model_answer, label):
            return label_score
    return 0.0

def contain_python_reward_fn(generated_text, golden_answer, task="math", is_r1_style=False):
    if "```python" in generated_text.lower():
        return 1.0
    else:
        return 0.0

def contain_python_wo_backticks_reward_fn(generated_text, task="math", is_r1_style=False):
    if "python" in generated_text.lower():
        return 1.0
    else:
        return 0.0

# ------------------------------------------------------------
# import math

# def match_answer(response):
#     is_matched = False
#     ans_marker = 'The answer is: '
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]
            
#     ans_marker = 'answer:\n'
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]

#     ans_marker = 'answer: '
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]

#     # Find boxed
#     ans_boxed = _last_boxed_only_string(response)
#     if ans_boxed:
#         is_matched = True
#         response = ans_boxed

#     # Grade
#     return is_matched, response


# def _last_boxed_only_string(string):
#     idx = string.rfind("\\boxed")
#     if idx < 0:
#         idx = string.rfind("\\fbox")
#         if idx < 0:
#             return None

#     i = idx
#     left_brace_idx = None
#     right_brace_idx = None
#     num_left_braces_open = 0
#     while i < len(string):
#         if string[i] == "{":
#             num_left_braces_open += 1
#             if left_brace_idx is None:
#                 left_brace_idx = i
#         elif string[i] == "}":
#             num_left_braces_open -= 1
#             if num_left_braces_open == 0:
#                 right_brace_idx = i
#                 break

#         i += 1

#     if left_brace_idx is None or right_brace_idx is None:
#         return None

#     return string[left_brace_idx + 1: right_brace_idx].strip()

# def qwen_reward_fn(generated_text, golden_answer, task="math"):
#     if golden_answer in ["A", "B", "C", "D"]:
#         task = "gpqa"
#         model_answer = extract_answer(generated_text, task)
#         accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5
#         # if "boxed" not in generated_text:
#         #     accuracy = -1.0
#         return accuracy

#     answer = golden_answer.lstrip('0') 
#     is_matched, model_output = match_answer(generated_text)
#     model_output = model_output.strip("The final answer is ").strip(". I hope it is correct.")
#     try:
#         if "\pi" in model_output or "\pi" in golden_answer:
#             equivs = []
#             for pi in [math.pi, 3.14]:
#                 equivs.append(math_equal(model_output, answer, timeout=True, pi=pi))
#             equiv = any(equivs)
#         else:
#             equiv = math_equal(model_output, answer, timeout=True)
#     except:
#         equiv = False

#     if equiv:
#         return 1.0
#     else:
#         return 0.0

# ------------------------------------------------------------


def majority_vote(
    solutions: List[str],
    ground_truth: str,
    task="math"
):
    model_answers = [extract_answer(generated_text, task) for generated_text in solutions]
    model_answers = [answer for answer in model_answers if answer is not None]

    if len(model_answers) == 0:
        return -0.5
    
    counter = Counter(model_answers)
    
    majority_answer, _ = counter.most_common(1)[0]
    accuracy = 1.0 if grade_answer(majority_answer, ground_truth) else -0.5
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    return accuracy

def test_time_train(
    solutions: List[str],
    ground_truth: List[str],
    task="math"):
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(set(ground_truth)) == 1
    
    if ground_truth[0] in ["A", "B", "C", "D"]:
        task = "gpqa"

    model_answers = [extract_answer(generated_text, task) for generated_text in solutions]
    model_answers = [answer for answer in model_answers if answer is not None]
    counter = Counter(model_answers)
    
    estimated_label, majority_count = counter.most_common(1)[0]

    rewards = [float(grade_answer(estimated_label, model_answer)) for model_answer in model_answers]
    # if majority_count / len(solutions) >= 0.5:
    #     rewards = [float(grade_answer(estimated_label, model_answer)) for model_answer in model_answers]
    # else:
    #     rewards = [0.0] * len(solutions)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"
    
    return rewards


def test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    task="math"):
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    if len(set(ground_truth)) != 1:
        print(f"Ground truth is not unique: {ground_truth}")
    assert len(set(ground_truth)) == 1
    ground_truth = ground_truth[0]
    
    model_answers = [extract_answer(generated_text, task) for generated_text in solutions]
    model_answers = [answer for answer in model_answers if answer is not None]
    counter = Counter(model_answers)
    
    estimated_label, majority_count = counter.most_common(1)[0]
    
    hit_rate = 1.0 if grade_answer(estimated_label, ground_truth) else 0.0
    majority_ratio = majority_count / len(solutions)
    true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    rewards = [float(grade_answer(model_answer, estimated_label)) for model_answer in model_answers]
    
    true_rewards = [float(grade_answer(model_answer, ground_truth)) for model_answer in model_answers]
    
    rewards_hit_rate = 0
    for reward, true_reward in zip(rewards, true_rewards):
        if reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(rewards)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"
    
    return hit_rate, rewards_hit_rate, majority_ratio, true_label_ratio, rewards


def qwen_math_equal_subprocess(prediction, reference, timeout_seconds=10):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(qwen_math_equal, prediction=prediction, reference=reference, timeout=False)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return False

def simplerl_reward_fn(generated_text, golden_answer):
    model_answer = extract_answer(generated_text, "math")
    accuracy = 1.0 if qwen_math_equal_subprocess(prediction=model_answer, reference=golden_answer) else -0.5
    if "boxed" not in generated_text:
        accuracy = -1.0
    return accuracy