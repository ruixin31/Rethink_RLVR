import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Union
from collections import Counter

import math
import numpy as np
import ray
import torch
from vllm import SamplingParams

from ttrl.models.model_utils import process_sequences
from ttrl.helper.logging_utils import init_logger
from ttrl.verifier.auto_verify import auto_verify
from ttrl.env.naive_samples_maker import Samples, NaiveSamplesMaker
from ttrl.verifier.qwen.qwen_eval import test_time_train_metrics

from ttrl.helper.utils import to

logger = init_logger(__name__)

class TTTSamplesMaker(NaiveSamplesMaker):

    @torch.no_grad()
    def evaluate_samples(self, eval_data: Union[List[str], dict], **kwargs):
        args = self.strategy.args
        # sampling_params = SamplingParams(
        #     temperature=kwargs.get("eval_temperature", 0.6),
        #     top_p=kwargs.get("top_p", 0.95),
        #     top_k=kwargs.get("top_k", -1),
        #     max_tokens=kwargs.get("max_new_tokens", 3072),
        #     min_tokens=kwargs.get("min_new_tokens", 16),
        #     skip_special_tokens=kwargs.get("skip_special_tokens", False),
        # )
        sampling_params = SamplingParams(
            temperature=kwargs.get("eval_temperature", 0.0),
            top_p=kwargs.get("eval_top_p", 1.0),
            max_tokens=kwargs.get("max_new_tokens", 3072)
        )
        print("TTT Sampling Params:", sampling_params)

        all_prompts, all_labels, all_indices = eval_data[
            "prompt"], eval_data["label"], eval_data["indice"]

        # if kwargs.get("eval_temperature", 0.0):
        #     # Distribute requests to engines and collect responses to outputs
        #     all_output_refs = []
        #     batch_size = (len(all_prompts) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
        #     for i, llm in enumerate(self.vllm_engines):
        #         prompts = all_prompts[i * batch_size: (i + 1) * batch_size]
        #         if prompts:
        #             all_output_refs.append(
        #                 llm.generate.remote(
        #                     sampling_params=sampling_params, prompts=prompts)
        #             )

        #     # Retrieve and combine results from all outputs
        #     all_outputs = sum(ray.get(all_output_refs), [])
        # else:
        #     all_output_refs = []
        #     # we generate multiple outputs for each prompt for stable evaluation
        #     for llm in self.vllm_engines:
        #         all_output_ref = llm.generate.remote(
        #             sampling_params=sampling_params, prompts=all_prompts)
        #         all_output_refs.append(all_output_ref)
            
        #     all_outputs = ray.get(all_output_refs)

        all_output_refs = []
        # we generate multiple outputs for each prompt for stable evaluation
        for llm in self.vllm_engines:
            all_output_ref = llm.generate.remote(
                sampling_params=sampling_params, prompts=all_prompts)
            all_output_refs.append(all_output_ref)
        
        all_outputs = ray.get(all_output_refs)

        all_accuracies = []
        verify_task = getattr(args, "verify_task_eval", args.verify_task)
        print(f"Using verification task: {verify_task}")
        print(len(all_outputs))
        for outputs in all_outputs:
            all_accuracies.append(auto_verify(verify_task, 1, outputs, all_labels))

        # print(all_accuracies)
        accuracy = np.mean([np.mean(acc) for acc in all_accuracies])

        metadata = []
        for prompt, label, indice in zip(all_prompts, all_labels, all_indices):
            metadata.append({"prompt": prompt, "label": label,
                            "indice": indice, "outputs": []})

        for outputs in all_outputs:
            for idx, output in enumerate(outputs):
                metadata[idx]["outputs"].append(output.outputs[0].text)

        # print all_outputs
        # print("Start printing all outputs")
        # import os
        # import json
        # data = []
        # for i in range(len(all_prompts)):
        #     prompt = metadata[i]["prompt"]
        #     label = metadata[i]["label"]
        #     outputs = metadata[i]["outputs"]
        #     if i < 1:
        #         print(f"Question {i+1}:", prompt)
        #         print(f"Answer {i+1}:", outputs)
        #         print(f"Label {i+1}:", label)
        #         print("\n\n")
        #     data.append({"question": prompt, "outputs": outputs, "label": label})
        # if not os.path.exists("all_outputs.json"):
        #     with open("all_outputs.json", "w") as f:
        #         json.dump(data, f)

        return {"accuracy": accuracy, "metadata": metadata}

    @torch.no_grad()
    def generate_samples(self, all_prompts: Union[List[str], dict], rank=0, world_size=8, **kwargs) -> List[Samples]:
        """
        Generate samples and return a list of Samples.
        """
        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            logprobs=kwargs.get("logprobs", 5),
            include_stop_str_in_output=True,
        )

        all_prompts, all_labels = all_prompts["prompt"], all_prompts["label"]
        
        pre_n_samples_per_prompt = args.n_samples_per_prompt
        if args.n_votes_per_prompt:
            m = args.n_samples_per_prompt
            n = args.n_votes_per_prompt
            args.n_samples_per_prompt = args.n_votes_per_prompt
        
        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum(
            [[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum(
            [[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(
            all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i *
                                                    batch_size: (i + 1) * batch_size]
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(
                        sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])
        samples_list = []

        for i in random.sample(list(range(len(all_prompts))), k=min(3, len(all_prompts))):
            print(f"Question {i+1}:", all_prompts[i])
            print(f"Answer {i+1}:", all_outputs[i].outputs[0].text)
            print("\n\n")
        
        # Compute entropy of all outputs
        all_logprobs = [output.outputs[0].logprobs for output in all_outputs]
        all_avg_token_entropies = []
        for logprobs in all_logprobs:
            entropies = []
            for step in logprobs:
                top_k_probs = []
                for token, prob in step.items():
                    top_k_probs.append(np.exp(prob.logprob))
                top_k_probs = np.array(top_k_probs)
                entropy = -np.sum(top_k_probs * np.log(top_k_probs))
                entropies.append(entropy)
            avg_token_entropy = np.mean(entropies) if entropies else 0
            all_avg_token_entropies.append(avg_token_entropy)
        print(f"Average token entropy: {np.mean(all_avg_token_entropies)}")

        # Add metrics and sampling
        all_pred_outputs = [output.outputs[0].text for output in all_outputs]
        n_prompts = len(all_pred_outputs) // args.n_samples_per_prompt
        # Add metrics
        all_hit_rates, all_rewards_hit_rates, all_majority_ratios, all_true_label_ratios = [], [], [], []
        for prompt_idx in range(n_prompts):
            group_pred_outputs = all_pred_outputs[args.n_samples_per_prompt *
                                        prompt_idx:args.n_samples_per_prompt*(prompt_idx+1)]
            group_labels = all_labels[args.n_samples_per_prompt *
                                      prompt_idx:args.n_samples_per_prompt*(prompt_idx+1)]
            # test_time_train_metrics
            hit_rate, rewards_hit_rate, majority_ratio, true_label_ratio, _ = test_time_train_metrics(group_pred_outputs, group_labels)
            all_hit_rates.append(hit_rate)
            all_rewards_hit_rates.append(rewards_hit_rate)
            all_majority_ratios.append(majority_ratio)
            all_true_label_ratios.append(true_label_ratio)
        print(f"Hit rate: {np.mean(all_hit_rates)}, Rewards Hit Rate: {np.mean(all_rewards_hit_rates)}, Majority Ratio: {np.mean(all_majority_ratios)}, True Label Ratio: {np.mean(all_true_label_ratios)}")
        ttt_metrics = {
            "hit_rate": np.mean(all_hit_rates),
            "rewards_hit_rate": np.mean(all_rewards_hit_rates),
            "majority_ratio": np.mean(all_majority_ratios),
            "true_label_ratio": np.mean(all_true_label_ratios),
            "entropy": np.mean(all_avg_token_entropies)
        }
        
        # Only perform sampling if we're using voting
        if args.n_votes_per_prompt and pre_n_samples_per_prompt < args.n_samples_per_prompt:
            # Sampling to pre_n_samples_per_prompt
            new_all_outputs, new_all_pred_outputs, new_all_labels, new_all_entropies = [], [], [], []
            # Random sampling
            for prompt_idx in range(n_prompts):
                start_idx = args.n_samples_per_prompt * prompt_idx
                end_idx = args.n_samples_per_prompt * (prompt_idx + 1)
                group_outputs = all_outputs[start_idx:end_idx]
                group_pred_outputs = all_pred_outputs[start_idx:end_idx]
                group_labels = all_labels[start_idx:end_idx]
                
                new_all_outputs.extend(group_outputs[:pre_n_samples_per_prompt])
                new_all_pred_outputs.extend(group_pred_outputs[:pre_n_samples_per_prompt])
                new_all_labels.extend(group_labels[:pre_n_samples_per_prompt])
                new_all_entropies.extend(all_avg_token_entropies[start_idx:end_idx][:pre_n_samples_per_prompt])

            # Entropy-based sampling
            # for prompt_idx in range(n_prompts):
            #     start_idx = args.n_samples_per_prompt * prompt_idx
            #     end_idx = args.n_samples_per_prompt * (prompt_idx + 1)
                
            #     group_outputs = all_outputs[start_idx:end_idx]
            #     group_pred_outputs = all_pred_outputs[start_idx:end_idx]
            #     group_labels = all_labels[start_idx:end_idx]
            #     group_entropies = all_avg_token_entropies[start_idx:end_idx]
            #     # Get rewards for this group
            #     _, _, _, _, rewards = test_time_train_metrics(group_pred_outputs, group_labels)
                
            #     # Up to n_correct samples with lowest entropy from correct answers
            #     correct_indices = [i for i, r in enumerate(rewards) if r > 0]
            #     n_correct = math.ceil(len(correct_indices) * pre_n_samples_per_prompt / args.n_samples_per_prompt)
            #     if correct_indices:
            #         correct_entropies = [group_entropies[i] for i in correct_indices]
            #         sorted_correct_indices = [correct_indices[i] for i in np.argsort(correct_entropies)[:n_correct]]
            #     else:
            #         sorted_correct_indices = []
                
            #     # n_left samples with highest entropy from incorrect answers
            #     n_left = pre_n_samples_per_prompt - len(sorted_correct_indices)
            #     wrong_indices = [i for i, r in enumerate(rewards) if r <= 0]
            #     if wrong_indices and n_left > 0:
            #         wrong_entropies = [group_entropies[i] for i in wrong_indices]
            #         sorted_wrong_indices = [wrong_indices[i] for i in np.argsort(wrong_entropies)[::-1][:n_left]]
            #     else:
            #         sorted_wrong_indices = []
                
            #     selected_indices = sorted_correct_indices + sorted_wrong_indices
                
            #     # If we still don't have enough samples, random sample the rest
            #     if len(selected_indices) < pre_n_samples_per_prompt:
            #         left_indices = [i for i in range(len(rewards)) if i not in selected_indices]
            #         if left_indices:
            #             random_indices = random.sample(left_indices, pre_n_samples_per_prompt - len(selected_indices))
            #             selected_indices += random_indices
                
            #     assert len(selected_indices) == pre_n_samples_per_prompt

            #     # Add selected samples to new lists
            #     new_all_outputs.extend([group_outputs[i] for i in selected_indices])
            #     new_all_pred_outputs.extend([group_pred_outputs[i] for i in selected_indices])
            #     new_all_labels.extend([group_labels[i] for i in selected_indices])
            #     new_all_entropies.extend([group_entropies[i] for i in selected_indices])
            
            all_outputs = new_all_outputs
            all_pred_outputs = new_all_pred_outputs
            all_labels = new_all_labels
            all_avg_token_entropies = new_all_entropies
            # recalculate_ttt_metrics
            ttt_metrics["post_entropy"] = np.mean(all_avg_token_entropies)
            all_rewards_hit_rates, all_true_label_ratios = [], []
            for prompt_idx in range(n_prompts):
                start_idx = pre_n_samples_per_prompt * prompt_idx
                end_idx = pre_n_samples_per_prompt * (prompt_idx + 1)
                
                group_pred_outputs = all_pred_outputs[start_idx:end_idx]
                group_labels = all_labels[start_idx:end_idx]
                _, rewards_hit_rate, _, true_label_ratio, _ = test_time_train_metrics(group_pred_outputs, group_labels)
                all_rewards_hit_rates.append(rewards_hit_rate)
                all_true_label_ratios.append(true_label_ratio)
            ttt_metrics["post_rewards_hit_rate"] = np.mean(all_rewards_hit_rates)
            ttt_metrics["post_true_label_ratio"] = np.mean(all_true_label_ratios)
        
        # Restore original n_samples_per_prompt
        args.n_samples_per_prompt = pre_n_samples_per_prompt
        
        assert all_pred_outputs == [output.outputs[0].text for output in all_outputs]
        all_pred_outputs = [output.outputs[0].text for output in all_outputs]

        all_pred_labels = auto_verify(
            args.verify_task, args.n_samples_per_prompt, all_outputs, all_labels)

        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i: i +
                                  self.strategy.args.micro_rollout_batch_size]
            pred_labels = all_pred_labels[i: i +
                                          self.strategy.args.micro_rollout_batch_size]

            # filter all negative samples
            # if self.strategy.args.filter_samples_by_reward and sum(pred_labels) <= 0:
            #     continue

            pred_labels = torch.tensor(
                pred_labels, device="cpu", dtype=torch.float)

            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(
                        max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(
                        output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [
                        pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(
                        output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    if output_ids[output_len - 1] != eos_token_id:
                        output_ids[min(output_len, len(
                            output_ids) - 1)] = eos_token_id

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cpu")
                attention_mask = attention_mask.to("cpu")
                action_mask = action_mask.to("cpu")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        labels=pred_labels,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids +
                                     list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))
                sequences = torch.tensor(sequences, device="cpu").unsqueeze(0)
                attention_mask = torch.tensor(
                    attention_mask, device="cpu").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(
                    num_actions, device="cpu", dtype=torch.float)
                total_length = torch.tensor(
                    packed_seq_lens, device="cpu", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        labels=pred_labels,
                    )
                )
        return samples_list, all_prompts, all_pred_outputs, all_pred_labels, ttt_metrics
