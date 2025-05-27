import srsly
import random
import os
import itertools
from abc import ABC
import math
from collections import defaultdict
import numpy as np
from typing import List

import torch
from tqdm import tqdm
from datasets import Dataset
import ray
from dataclasses import dataclass
from ray.util.placement_group import placement_group

from ttrl.trainer.ppo_actor import ActorModelRayActor
from ttrl.trainer.ppo_critic import CriticModelRayActor
from ttrl.trainer.ppo_prime import PrimeModelRayActor
from ttrl.models.vllm_engine import create_vllm_engines
from ttrl.models.ray_launcher import PPORayActorGroup, ReferenceModelRayActor, RewardModelRayActor
from ttrl.datasets.prompts_dataset import PromptDatasetWithLabel
from ttrl.datasets.sft_dataset import SFTDataset
from ttrl.helper.utils import blending_datasets, get_tokenizer
from ttrl.helper.distributed_sampler import ResumableRandomSampler

from ttrl.env.prime_samples_maker import PrimeSamplesMaker
from ttrl.models.model_utils import unpacking_samples


@dataclass
class Agent:
    # One agent includes actor model (many workers), critic model, reference model, and vllm engines for PPO training
    # The agent can start training after making trajectory samples by itself or with other agents
    actor_model_group: PPORayActorGroup
    critic_model_group: PPORayActorGroup
    vllm_engines: List

    def save_actor_and_critic_model(self, args, tokenizer):
        ray.get(self.actor_model_group.async_save_model())
        if args.critic_pretrain and args.save_value_network:
            ray.get(self.critic_model_group.async_save_model())

        # save tokenizer
        tokenizer.save_pretrained(args.save_path)


@dataclass
class Prime:
    prime_model_group: PPORayActorGroup
    prime_ref_group: PPORayActorGroup


@ray.remote
def generate_samples_remote(samples_maker, chunk_prompts, rank, world_size, generate_kwargs):
    samples_list, all_prompts, all_pred_outputs, all_pred_labels = samples_maker.generate_samples(
        chunk_prompts, rank, world_size, **generate_kwargs)
    return {
        "sample": samples_list,
        "prompt": all_prompts,
        "output": all_pred_outputs,
        "label": all_pred_labels
    }


class PrimeController(ABC):
    """
    Load prompt datasets
    Manage experience_maker
    Run global fit function (TODO: use MLFlow)
    Log status for each fit
        including various actors/critics
    """

    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
        args = strategy.args
        self.args = args

        self.tokenizer = get_tokenizer(
            args.pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)

        self.generate_kwargs = {
            "do_sample": True,
            "max_new_tokens": args.generate_max_len,
            "max_length": args.max_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # prepare_datasets
        self.prepare_datasets()
        self.num_update_steps_per_episodes = (
            len(self.prompts_dataset) *
            args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
        )
        max_steps = math.ceil(args.num_episodes *
                              self.num_update_steps_per_episodes)
        self._max_steps = max_steps

        # TODO: init logging, add MLFlow
        self._init_logging(strategy)

    def build_env(self):
        # prepare agent includes many workers
        self.agent = self._init_agent()

        # create samples maker
        self.samples_maker = PrimeSamplesMaker(
            strategy=self.strategy, tokenizer=self.tokenizer, vllm_engines=self.agent.vllm_engines)

        self.prime_model = self._init_prime()

    def _init_prime(self):
        args = self.args
        strategy = self.strategy
        # if colocated, create placement group for actor and ref model explicitly.
        pg = None
        if args.colocate_prime_ref:
            assert (
                args.prime_num_nodes == args.prime_ref_num_nodes and args.prime_num_gpus_per_node == args.prime_ref_num_gpus_per_node
            ), "num_nodes and num_gpus_per_node must be the same when colocate prime and ref model."

            bundles = [
                {"GPU": args.prime_num_gpus_per_node,
                    "CPU": args.prime_num_gpus_per_node}
                for _ in range(args.prime_num_nodes)
            ]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())
        prime_model = PPORayActorGroup(
            args.prime_num_nodes,
            args.prime_num_gpus_per_node,
            PrimeModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.75 if pg else 1,
        )

        prime_ref_model = PPORayActorGroup(
            args.prime_ref_num_nodes,
            args.prime_ref_num_gpus_per_node,
            ReferenceModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.25 if pg else 1,
        )

        refs = []
        refs.extend(prime_ref_model.async_init_model_from_pretrained(
            strategy, args.prime_pretrain))
        refs.extend(prime_model.async_init_model_from_pretrained(
            strategy, args.prime_pretrain, self._max_steps))
        ray.get(refs)

        # init actor and critic mdoel
        refs = prime_model.async_init_prime_trainer(
            prime_ref_model
        )
        ray.get(refs)

        return prime_model

    def _init_agent(self):
        args = self.args
        strategy = self.strategy
        # if colocated, create placement group for actor and ref model explicitly.
        pg = None
        if args.colocate_actor_ref:
            assert (
                args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

            bundles = [
                {"GPU": args.actor_num_gpus_per_node,
                    "CPU": args.actor_num_gpus_per_node}
                for _ in range(args.actor_num_nodes)
            ]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())

        # NOTE(wuxibin): Why don't we allocate 0.5 gpu for each actor when colocate models?
        # Say we have 1 node with 4 GPUs, and num_gpus_per_node for each model is 4.
        # If we allocate 0.5 gpu for both actor and ref model, then gpu allocation is
        #   |actor|actor|actor|actor|  ref | ref  | ref  | ref |
        #   |GPU0 |GPU0 |GPU1 |GPU1 | GPU2 | GPU2 | GPU3 | GPU3 |
        #
        # So 0.75/0.25 gpu is a tricky to let Ray spread all models evenly on all gpus.
        #   |actor| ref  |actor| ref  |actor| ref  |actor|ref  |
        #   |GPU0 | GPU0 |GPU1 | GPU1 |GPU2 | GPU2 |GPU3 | GPU3 |
        actor_model = PPORayActorGroup(
            args.actor_num_nodes,
            args.actor_num_gpus_per_node,
            ActorModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.75 if pg else 1,
        )

        ref_model = PPORayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            ReferenceModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.25 if pg else 1,
        )

        # if colocated, create placement group for critic and reward model explicitly.
        pg = None
        if args.critic_pretrain and args.colocate_critic_reward:
            assert (
                args.critic_num_nodes == args.reward_num_nodes
                and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
            ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

            bundles = [
                {"GPU": args.critic_num_gpus_per_node,
                    "CPU": args.critic_num_gpus_per_node}
                for _ in range(args.critic_num_nodes)
            ]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())

        if args.critic_pretrain:
            critic_model = PPORayActorGroup(
                args.critic_num_nodes,
                args.critic_num_gpus_per_node,
                CriticModelRayActor,
                pg=pg,
                num_gpus_per_actor=0.75 if pg else 1,
            )
        else:
            critic_model = None

        # multiple reward models
        if args.reward_pretrain is not None:
            reward_pretrains = args.reward_pretrain.split(",")
            reward_models = []
            for _ in reward_pretrains:
                reward_models.append(
                    PPORayActorGroup(
                        args.reward_num_nodes,
                        args.reward_num_gpus_per_node,
                        RewardModelRayActor,
                        pg=pg,
                        num_gpus_per_actor=0.25 if pg else 1,
                    )
                )
        else:
            reward_models = None

        # init reference/reward/actor model
        refs = []
        refs.extend(ref_model.async_init_model_from_pretrained(
            strategy, args.pretrain))
        refs.extend(actor_model.async_init_model_from_pretrained(
            strategy, args.pretrain, self._max_steps))
        if args.reward_pretrain is not None:
            for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                refs.extend(reward_model.async_init_model_from_pretrained(
                    strategy, reward_pretrain))

        ray.get(refs)

        # init vLLM engine for text generation
        vllm_engines = None
        if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
            max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            vllm_engines = create_vllm_engines(
                args.vllm_num_engines,
                args.vllm_tensor_parallel_size,
                args.pretrain,
                args.seed,
                args.enable_prefix_caching,
                args.enforce_eager,
                max_len,
            )

        if args.critic_pretrain:
            # critic scheduler initialization depends on max_step, so we have to init critic after actor
            # TODO: use first reward model as critic model
            refs.extend(critic_model.async_init_model_from_pretrained(
                strategy, args.critic_pretrain, self._max_steps))
            ray.get(refs)

        # init actor and critic mdoel
        refs = actor_model.async_init_actor_trainer(
            critic_model, ref_model, reward_models, args.remote_rm_url, vllm_engines=vllm_engines
        )
        ray.get(refs)

        agent = Agent(
            actor_model_group=actor_model,
            critic_model_group=critic_model,
            vllm_engines=vllm_engines,
        )

        return agent

    def _init_logging(self, strategy):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb:
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
                settings=wandb.Settings(init_timeout=240)

            wandb.define_metric("train/global_step")
            wandb.define_metric(
                "train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric(
                "eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(
                self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        prompts_data, prompts_data_eval = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=True,
            train_split=args.prompt_split,
        )
        prompts_data = prompts_data.select(
            range(min(args.max_samples, len(prompts_data))))
        self.prompts_dataset = PromptDatasetWithLabel(
            prompts_data, self.tokenizer, strategy, input_template=args.input_template, add_prompt_suffix=args.add_prompt_suffix
        )
        self.prompts_dataset_eval = PromptDatasetWithLabel(
            prompts_data_eval, self.tokenizer, strategy, input_template=args.input_template, add_prompt_suffix=args.add_prompt_suffix
        )
        sampler = ResumableRandomSampler(
            data_source=self.prompts_dataset,
            batch_size=args.rollout_batch_size,
            drop_last=True,
            shuffle=True,
            seed=args.seed
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset, args.rollout_batch_size, True, True,
            sampler=sampler
        )

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
                train_split=args.pretrain_split,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + \
                args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(
                    range(
                        min(
                            len(pretrain_data), args.max_epochs *
                            len(self.prompts_dataset) *
                            args.n_samples_per_prompt
                        )
                    )
                ),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def load_checkpoint_steps(self):
        args = self.args
        num_update_steps_per_episodes = (
            len(self.prompts_dataset) *
            args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
        )
        consumed_samples = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            states = torch.load(os.path.join(ckpt_path, "step_states.bin"))
            consumed_samples = states["consumed_samples"]
            print(
                f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {consumed_samples}")

        return num_update_steps_per_episodes, consumed_samples

    def run(self):
        # update steps if load checkpoints
        num_update_steps_per_episodes, consumed_samples = self.load_checkpoint_steps()

        # start fitting
        self.fit(
            consumed_samples=consumed_samples,
            num_update_steps_per_episodes=num_update_steps_per_episodes
        )

        # save actor and critic workers in agent
        self.agent.save_actor_and_critic_model(self.args, self.tokenizer)
        self.prime_model.async_save_model()

    def clean_template(self, prompt):
        """
        clean the template
        '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nhow are you?<|im_end|>\n<|im_start|>assistant\n'

        '<｜begin▁of▁sentence｜><｜User｜>how are you?<｜Assistant｜>'
        """
        if "<|im_start|>" in prompt:
            prompt = prompt.split(
                "<|im_start|>user\n")[-1].split("<|im_end|>")[0]
        elif "<｜begin▁of▁sentence｜>" in prompt:
            prompt = prompt.split(
                "<｜begin▁of▁sentence｜><｜User｜>")[-1].split("<｜Assistant｜>")[0]
        return prompt

    def build_sft_sample_list(self, data):
        """
        Input: {
        "sample": samples_list,
        "prompt": all_prompts,
        "output": all_pred_outputs,
        "label": all_pred_labels
        }

        Output: List[dict]
            {input_key: "", label_key:""}
        """
        all_prompts = data["prompt"]
        all_outputs = data["output"]
        all_labels = data["label"]
        assert len(all_prompts) == len(all_outputs) == len(all_labels)
        n_samples_per_prompt = self.args.n_samples_per_prompt
        sample_list = []
        for idx in range(0, len(all_prompts), n_samples_per_prompt):
            tmp_sample = []
            for i in range(n_samples_per_prompt):
                prompt = all_prompts[idx + i]
                output = all_outputs[idx + i]
                label = all_labels[idx + i]
                if label == 1:
                    prompt = self.clean_template(prompt)
                    tmp_sample.append({
                        self.args.input_key: prompt,
                        self.args.output_key: output,
                    })
            if len(tmp_sample) <= n_samples_per_prompt // 2:
                # sample_list.append(random.choice(tmp_sample))
                sample_list.extend(tmp_sample)
        return sample_list

    def build_sft_dataset(self, data_list):
        pretrain_data = Dataset.from_list(data_list)
        pretrain_max_len = self.strategy.args.max_len if self.strategy.args.max_len else self.strategy.args.prompt_max_len + \
            self.strategy.args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data,
            self.tokenizer,
            pretrain_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1
        )
        assert len(
            pretrain_dataset) > 0, f"{len(pretrain_dataset)} samples are generated."
        return pretrain_dataset

    def generate_shared_samples(self, steps, rand_prompts):
        world_size = self.agent.actor_model_group.world_size

        any_key = next(iter(rand_prompts.keys()))
        length = len(rand_prompts[any_key])
        chunk_size = (length + world_size - 1) // world_size
        chunked = [dict() for _ in range(world_size)]
        for key, data_list in rand_prompts.items():
            for i in range(world_size):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, length)
                sub_slice = data_list[start_idx:end_idx]
                chunked[i][key] = sub_slice

        all_refs = []
        for rank in range(world_size):
            samples_ref = generate_samples_remote.remote(
                self.samples_maker, chunked[rank], rank, world_size, self.generate_kwargs)
            all_refs.append(samples_ref)
        all_results = sum([r["sample"] for r in ray.get(all_refs)], [])

        # TODO: training prime and compute implicit rewards
        # 1) firstly, compute log_probs_ref and save in samples
        # 2) then, training prime model with samples
        # 3) later,compute implicit rewards and save in samples
        # 4) finally, pass samples to actor models
        all_results_with_rewards_refs = self.prime_model.async_fit_and_reward_prime_model(
            steps, all_results)
        all_results = ray.get(all_results_with_rewards_refs)
        all_results_with_rewards = [a[0] for a in all_results]
        prime_status = [a[1] for a in all_results if a[1]["is_rank_0"]][0]
        del prime_status["is_rank_0"]
        prime_status["prime/token_score"] = self.compute_average_rewards(
            all_results_with_rewards)

        sharded_data_refs = [None for _ in range(world_size)]
        sft_samples = []
        assert len(
            all_results_with_rewards) == world_size, f"{len(all_results_with_rewards)} != {world_size}"

        # create samples for each actor worker
        for rank in range(world_size):
            shard_data = all_results_with_rewards[rank]
            shard_ref = ray.put(shard_data)
            if self.args.training_mode in ["sft", "both", "mix"]:
                sft_samples.extend(self.build_sft_sample_list(shard_data))

            sharded_data_refs[rank] = shard_ref

        if len(sft_samples) > 0:
            for sample in random.sample(sft_samples, 3):
                print(sample)
                print("="*20)
            return sharded_data_refs, ray.put(self.build_sft_dataset(sft_samples)), prime_status
        else:
            return sharded_data_refs, None, prime_status

    def compute_average_rewards(self, all_samples):
        all_rewards = []
        for samples in sum(all_samples, []):
            process_rewards = unpacking_samples(
                samples.process_rewards, samples.num_actions)
            all_rewards.extend([rewards.sum().item()
                               for rewards in process_rewards])
        return np.mean(all_rewards)

    def fit(self,
            consumed_samples=0,
            num_update_steps_per_episodes=1,
            ) -> None:
        args = self.args

        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (
            num_rollouts_per_episodes * args.rollout_batch_size)

        # Eval before training
        acc = self.evaluate_samples(steps)
        self.save_logs(args, steps, {"eval/reward": acc}, None)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, ResumableRandomSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]"
            )

            # pass generated samples to each actor worker
            for rand_prompts in self.prompts_dataloader:

                # make shared samples refs
                shared_data_refs, sft_dataset, prime_status = self.generate_shared_samples(steps,
                                                                                           rand_prompts=rand_prompts)

                # start training actor models (workers)
                refs = self.agent.actor_model_group.async_fit_actor_model(
                    steps, shared_data_refs, sft_dataset)
                results = ray.get(refs)

                # find result from rank 0
                for result in results:
                    logs_dict, is_rank_0, perf_stats = result["status"], result["is_rank_0"], result["perf_stats"]
                    if is_rank_0:
                        break

                logs_dict.update(prime_status)

                if steps % args.eval_steps == 0:
                    acc = self.evaluate_samples(steps)
                    logs_dict["eval/reward"] = acc

                self.save_logs(args, steps, logs_dict, perf_stats)

                pbar.set_postfix(logs_dict)
                pbar.update()
                steps = steps + 1

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()

    def list_of_dicts_to_dict_of_lists(self, list_of_dicts):
        result = defaultdict(list)
        for d in list_of_dicts:
            for key, value in d.items():
                result[key].append(value)
        return dict(result)

    def evaluate_samples(self, steps):
        data_list = self.prompts_dataset_eval.get_all_prompts()
        all_prompts = self.list_of_dicts_to_dict_of_lists(data_list)
        results = self.samples_maker.evaluate_samples(all_prompts)
        output_path = os.path.join(self.args.save_path, "eval_results")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        srsly.write_json(os.path.join(
            output_path, f"global_steps{steps}.json"), results)
        return results["accuracy"]

    def save_logs(self, args, global_step, logs_dict, perf_stats):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None:
                logs = {
                    ( k if "eval/" in k else f"train/{k}"): v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if perf_stats is not None:
                    logs.update(
                        {f"perf/experience_maker/{k}": v for k, v in perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    # save eval logs to eval folder
                    if len(k.split("/")) == 2:
                        split, k = k.split("/")
                    else:
                        split = "train"
                    self._tensorboard.add_scalar(
                        f"{split}/{k}", v, global_step)
                if perf_stats is not None:
                    for k, v in perf_stats.items():
                        self._tensorboard.add_scalar(
                            f"perf/experience_maker/{k}", v, global_step)
