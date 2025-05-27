import math
from itertools import groupby, accumulate

import os
import torch

from tqdm import tqdm
from typing import List
from torch.nn import functional as F

import ray
from transformers.trainer import get_scheduler

from ttrl.models.actor import Actor
from ttrl.trainer.ppo_trainer import PPOTrainer
from ttrl.helper.deepspeed import DeepspeedStrategy
from ttrl.trainer.utils_prime import PrimeSamples, PrimeSamplesDataset
from ttrl.models.ray_launcher import BasePPORole
from ttrl.models.model_utils import unpacking_samples


class PrimePPOTrainer(PPOTrainer):
    def __init__(self, *args,
                 prime_model: Actor,
                 prime_optim,
                 prime_scheduler,
                 prime_beta: float = 0.05,
                 prime_granularity: str = "token",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.prime_model = prime_model
        self.prime_optim = prime_optim
        self.prime_scheduler = prime_scheduler
        self.prime_beta = prime_beta
        self.prime_granularity = prime_granularity

        self.packing_samples = getattr(self.args, "packing_samples", False)

    def ppo_train(self, prime_samples: List[PrimeSamples]):
        prime_dataset = PrimeSamplesDataset(prime_samples)

        # get packed PrimeSamples
        dataloader = self.strategy.setup_dataloader(
            prime_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0]
        )

        device = torch.cuda.current_device()

        status_list = []
        status_mean = []
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0()
            )

            for samples in pbar:
                samples.to_device(device)
                
                if hasattr(self.args, "num_agents"):
                    assert len(samples.info) == self.args.num_agents, f"num_agents should be provided in args, but got {len(samples.info)} agents"
                
                samples.base_action_log_probs = samples.base_action_log_probs.to(device)
                # self.strategy.print(samples)
                status = self.train_step_prime(samples)

                status = self.strategy.all_reduce(status)

                status_list.append(status)
                pbar.set_postfix(status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def compute_ce_dpo_loss_rm(
        self,
        # (batch_size, seq_len), from (logp_policy - logp_ref)
        token_level_scores: torch.Tensor,
        # (batch_size), (0/1) or float (0./1.)
        preferences: torch.Tensor,
        action_mask: torch.Tensor = None,        # (batch_size, seq_len)
        num_actions=None
    ) -> torch.Tensor:
        # shape: [batch_size]
        if action_mask is not None:
            mask_scores = token_level_scores * action_mask
        else:
            mask_scores = token_level_scores

        if num_actions is not None:
            mask_scores = unpacking_samples(token_level_scores, num_actions)
            summed_scores = torch.stack([ms.sum() for ms in mask_scores])
        else:
            summed_scores = (mask_scores).sum(dim=1)

        # \sigma(\beta * sum)
        probs = (summed_scores * self.prime_beta).sigmoid()  # [batch_size]
        labels = (preferences == 1).to(probs.dtype)
        loss = F.binary_cross_entropy(probs, labels)
        return loss

    def train_step_prime(self, data: PrimeSamples):
        self.prime_model.train()
        if not self.packing_samples:
            sequences = torch.cat(data.sequences, dim=0).unsqueeze(0)
            base_action_log_probs = torch.cat(
                data.base_action_log_probs, dim=0).unsqueeze(0)
            attention_mask = torch.cat(data.attention_mask, dim=0).unsqueeze(0)
            action_mask = torch.cat(data.action_mask, dim=0).unsqueeze(0)
            num_actions = data.num_actions
            packed_seq_lens = data.packed_seq_lens
            # response_length = data.response_length
            labels = data.labels
        else:
            sequences = data.sequences
            base_action_log_probs = data.base_action_log_probs
            attention_mask = data.attention_mask
            action_mask = data.action_mask
            num_actions = data.num_actions
            packed_seq_lens = data.packed_seq_lens
            # response_length = data.response_length
            labels = data.labels

        # get log_probs for action parts
        policy_log_probs = self.prime_model(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=False,
            packed_seq_lens=packed_seq_lens
        )

        token_level_score, q = self.compute_implicit_reward(
            log_prob=policy_log_probs,
            ref_log_prob=base_action_log_probs,
            num_actions=num_actions,
            action_mask=action_mask
        )

        prime_loss = self.compute_ce_dpo_loss_rm(
            token_level_scores=q,
            preferences=labels,
            action_mask=None,  # alread get logprobs of actions in actor.forward()
            num_actions=num_actions
        )

        # nll loss
        # if not self.strategy.args.prime_nll_loss:
        #     nll_loss = 0

        loss = prime_loss  # + nll_loss * self.args.prime_nll_loss_coef
        self.strategy.backward(
            loss, self.prime_model, self.prime_optim)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.prime_model.model.parameters(), max_norm=self.strategy.args.prime_grad_clip)

        self.strategy.optimizer_step(
            self.prime_optim, self.prime_model, self.prime_scheduler)
        return {
            "prime/loss": prime_loss.item(),
            "prime/lr": self.prime_scheduler.get_last_lr()[0],
            "prime/grad_norm": grad_norm.item(),
        }

    def compute_implicit_reward(self, log_prob, ref_log_prob, num_actions, action_mask):
        if action_mask is not None:
            batch_action_mask = unpacking_samples(action_mask, num_actions)

        def get_ends(i, n_act):
            if self.prime_granularity == "token":
                return list(range(n_act))
            elif self.prime_granularity == "whole":
                return [n_act - 1]
            elif self.prime_granularity == "agent":
                # TODO: get agent-level rewards based on action_mask
                assert action_mask is not None, "action_mask should be provided while computing agent-level rewards"
                current_action_mask = batch_action_mask[i]
                counts = [len(list(group)) for _, group in groupby(current_action_mask)]

                return [x - 1 for x in accumulate(counts)]
            else:
                raise NotImplementedError

        q = log_prob - ref_log_prob
        token_level_score = torch.zeros_like(q).to(q.dtype)
        batch_size = len(num_actions)

        # reward computation does not need gradient. only q needs
        with torch.no_grad():
            if self.packing_samples:
                offset = 0
                for i in range(batch_size):
                    n_act = num_actions[i]
                    ends = get_ends(i, n_act)

                    # the strategy of translating q to reward function:
                    prev_end = 0
                    for end_idx in ends:
                        start, end = prev_end, end_idx
                        token_level_score[0, offset + end] = q[0,
                                                               offset + start: offset + end + 1].sum()
                        prev_end = end + 1

                    offset += n_act

            else:
                for b in range(batch_size):
                    n_act = num_actions[b]
                    ends = get_ends(b, n_act)
                    # the strategy of translating q to reward function:
                    prev_end = 0
                    for end_idx in ends:
                        start, end = prev_end, end_idx
                        token_level_score[b, end] = q[b, start: end + 1].sum()
                        prev_end = end + 1

        # From PRIME - this method will still consider the relative value of rewards.
        # The key is to control the absolute value of RETURN from being too high.
        # so the normalization is done by controlling the maximum of reverse cumulative sum
        if getattr(self.strategy.args, "batch_norm", False):
            if self.strategy.args.packing_samples:
                token_level_score = unpacking_samples(
                    token_level_score, num_actions)

            for i in range(len(token_level_score)):
                normalized_token_level_score = token_level_score[i]

                reverse_cumsum = torch.cumsum(
                    normalized_token_level_score.flip(dims=[0]), dim=0).flip(dims=[0])
                normalized_token_level_score = normalized_token_level_score / \
                    (reverse_cumsum.abs().max() + 1e-6)

                token_level_score[i] = normalized_token_level_score

            if self.strategy.args.packing_samples:
                token_level_score = torch.cat(
                    token_level_score, dim=0).unsqueeze(0)

        return token_level_score, q

    def compute_ref_log_probs_for_samples(self, prime_samples: List[PrimeSamples]):
        """
        Use `self.initial_model` to perform a forward pass only once for each `PrimeSamples`,
        and store the resulting `base_action_log_probs` in `prime_samples[i].base_action_log_probs`.
        """
        prime_dataset = PrimeSamplesDataset(prime_samples)
        dataloader = self.strategy.setup_dataloader(
            prime_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )

        for samples in dataloader:
            
            if hasattr(self.args, "num_agents"):
                assert len(samples.info) == self.args.num_agents, f"num_agents should be provided in args, but got {len(samples.info)} agents"
            
            sequences_cpu = samples.sequences.to("cpu")
            attention_mask_cpu = samples.attention_mask.to("cpu")
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu,
                samples.num_actions,
                attention_mask_cpu,
                packed_seq_lens=samples.packed_seq_lens
            )
            base_action_log_probs = ray.get([base_action_log_probs_ref])[0]
            if self.strategy.args.colocate_actor_ref:
                ray.get([self.initial_model.empty_cache.remote()])
            samples.base_action_log_probs = base_action_log_probs

        return prime_dataset

    def compute_final_rewards_for_samples(self, prime_samples: List[PrimeSamples]) -> List[PrimeSamples]:
        """
        After the prime_model training is completed, perform a forward pass on all prime_samples,
        calculate the reward (for example, a certain aggregation of q = log_prob - ref_log_prob), and store the results in samples.rewards.
        """
        device = torch.cuda.current_device()
        self.prime_model.eval()

        # Distributed Sampler for DeepSpeed
        prime_dataset = PrimeSamplesDataset(prime_samples)
        dataloader = self.strategy.setup_dataloader(
            prime_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )

        return_samples_list = []
        with torch.no_grad():
            for samples in dataloader:
                samples.to_device(device)
                samples.base_action_log_probs = samples.base_action_log_probs.to(
                    device)

                if hasattr(self.args, "num_agents"):
                    assert len(samples.info) == self.args.num_agents, f"num_agents should be provided in args, but got {len(samples.info)} agents"
                
                policy_log_probs, _ = self.prime_model(
                    samples.sequences,
                    samples.num_actions,
                    attention_mask=samples.attention_mask,
                    return_output=True,
                    packed_seq_lens=samples.packed_seq_lens
                )

                token_level_score, q = self.compute_implicit_reward(
                    log_prob=policy_log_probs,
                    ref_log_prob=samples.base_action_log_probs,
                    num_actions=samples.num_actions,
                    action_mask=samples.action_mask
                )

                # return q or token-level scores?
                samples.process_rewards = token_level_score

                return_samples_list.append(samples)
        return return_samples_list


@ray.remote(num_gpus=1)
class PrimeModelRayActor(BasePPORole):
    def init_model_from_pretrained(self,
                                   strategy: DeepspeedStrategy,
                                   pretrain,
                                   max_steps,
                                   rolename="prime"):
        self._setup_distributed(strategy)
        args = self.strategy.args
        self.args = args
        self.rolename = rolename

        prime_model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
        )

        # configure optimizer
        prime_optim = strategy.create_optimizer(
            prime_model, lr=args.prime_learning_rate, betas=strategy.args.prime_adam_betas, weight_decay=args.prime_l2
        )

        prime_scheduler = get_scheduler(
            getattr(args, "prime_scheduler", "cosine_with_min_lr"),
            prime_optim,
            num_warmup_steps=math.ceil(max_steps * args.prime_lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={
                "min_lr": args.prime_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            prime_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.prime_model, self.prime_optim, self.prime_scheduler = strategy.prepare(
            (prime_model, prime_optim, prime_scheduler),
            is_rlhf=True,
        )

    def init_trainer(self, initial_model: ray.actor.ActorHandle):
        strategy = self.strategy
        args = strategy.args
        self.trainer = PrimePPOTrainer(
            strategy,
            actor=None,
            critic=None,
            reward_model=None,
            initial_model=initial_model,
            ema_model=None,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=None,
            critic_scheduler=None,
            prime_model=self.prime_model,
            prime_optim=self.prime_optim,
            prime_scheduler=self.prime_scheduler,
            prime_beta=args.prime_beta,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            prime_granularity=args.prime_granularity,
            rolename=self.rolename
        )

    def fit(self, steps, prime_samples):
        torch.cuda.empty_cache()
        self.prime_model.train()
        status = self.trainer.ppo_train(prime_samples)
        torch.cuda.empty_cache()
        return status

    def fit_and_reward(self, steps, prime_samples: List[PrimeSamples]):
        """
        1) First, use the initial_model to calculate ref_log_probs and store them in prime_samples.
        2) Use prime_samples to train the prime_model.
        3) After training is complete, calculate the final reward and write it back to prime_samples.
        4) Return prime_samples with the reward.
        """
        self.empty_cache()
        self.trainer.compute_ref_log_probs_for_samples(prime_samples)
        self.empty_cache()
        status = self.trainer.ppo_train(prime_samples)
        for sample in prime_samples:
            sample.to_device("cpu")
        status["is_rank_0"] = self.strategy.is_rank_0()
        self.empty_cache()
        prime_samples = self.trainer.compute_final_rewards_for_samples(
            prime_samples)
        for samples in prime_samples:
            samples.to_device("cpu")

        self.empty_cache()
        self.save_checkpoint(steps)
        return prime_samples, status

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.prime_model,
            None,
            args.save_path + "_" + self.rolename,
        )

    def save_checkpoint(self, global_step):
        args = self.strategy.args
        if global_step % args.save_steps != 0:
            return

        tag = f"global_step{global_step}"

        self.strategy.save_ckpt(
            self.prime_model.model,
            os.path.join(args.ckpt_path, "_" + self.rolename),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem
        )
