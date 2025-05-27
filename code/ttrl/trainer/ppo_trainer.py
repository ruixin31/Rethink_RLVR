import os
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional
import socket
import deepspeed
import ray

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ttrl.models.actor import Actor
from ttrl.models.loss import GPTLMLoss, PolicyLoss, ValueLoss, PolicyLossWoOldLogProbs

from ttrl.models.model_utils import masked_mean
from ttrl.helper.distributed_sampler import DistributedSampler
from ttrl.helper.distributed_utils import init_process_group

from ttrl.trainer.experience_maker import Experience, ExperienceMaker
from ttrl.trainer.kl_controller import AdaptiveKLController, FixedKLController
from ttrl.trainer.replay_buffer import NaiveReplayBuffer


class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        entropy_coef: float = 0.0,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        pretrain_dataloader=None,
        rolename=None,
        **generate_kwargs,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler
        
        assert rolename is not None, "rolename should not be None"
        self.rolename = rolename
        
        self.actor_loss_fn = PolicyLoss(eps_clip)

        if self.args.use_policy_loss_wo_old_log_probs:
            self.actor_loss_fn = PolicyLossWoOldLogProbs(eps_clip)

        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        self.pretrain_dataloader = pretrain_dataloader

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)
        
        self.entropy_coef = entropy_coef

        self.experience_maker = ExperienceMaker(
            actor,
            critic,
            reward_model,
            initial_model,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
        )

        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples
        )

    def ppo_train(self, global_steps=0):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}] ({self.rolename})",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }
                
                if "gt_reward" in status:
                    short_status['gt_reward'] = status['gt_reward']

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                if "entropy_loss" in status:
                    short_status["entropy"] = status["entropy_loss"]

                status_list.append(status)

                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        pass
    
    def fit(self, steps, samples_ref):
        all_samples = samples_ref
        args = self.args
        
        if args.training_mode != "sft":
            for i, experience in enumerate(
                self.experience_maker.make_experience_list(all_samples, **self.generate_kwargs)
            ):
                self.replay_buffer.append(experience)
        
        torch.cuda.empty_cache()
        if args.training_mode != "sft":
            self.replay_buffer.normalize("advantages", self.strategy)
        status = self.ppo_train(steps)
        if args.training_mode != "sft":
            self.replay_buffer.clear()
        torch.cuda.empty_cache()

        if "kl" in status:
            self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

        # logs/checkpoints
        client_states = {"consumed_samples": steps * args.rollout_batch_size}
        self.save_checkpoints(args, steps, client_states)

        return {
            "status": status,
            "is_rank_0": self.strategy.is_rank_0(),
            "perf_stats": self.experience_maker.perf_stats
        }

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = experience.base_action_log_probs

    
        if self.args.use_entropy_loss:
            # actor loss
            action_log_probs, entropy, output = self.actor(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                return_output=True,
                packed_seq_lens=packed_seq_lens,
                return_entropy=True,
            )
        elif self.args.use_online_sft:
            action_log_probs, cross_entropy, output = self.actor(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                return_output=True,
                packed_seq_lens=packed_seq_lens,
                return_cross_entropy=True,
            )
        else:
            # actor loss
            action_log_probs, output = self.actor(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                return_output=True,
                packed_seq_lens=packed_seq_lens,
            )

        # loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
        )

        if self.args.use_kl_loss:
            kl_loss = action_log_probs - base_action_log_probs
            if self.args.use_kl_estimator_k3:
                kl_loss = -kl_loss
                r = kl_loss.exp()
                kl_loss = r - 1.0 - kl_loss
            kl_loss = masked_mean(kl_loss, experience.action_mask, dim=-1).mean()
        else:
            kl_loss = 0

        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        if self.args.use_entropy_loss:
            entropy_loss = masked_mean(entropy, experience.action_mask, dim=-1).mean()  
            loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value - entropy_loss * self.entropy_coef
        elif self.args.use_online_sft:
            cross_entropy = masked_mean(cross_entropy, experience.action_mask, dim=-1).mean()
            loss = cross_entropy + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value
        else:
            loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value
        self.strategy.backward(loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # status
        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()

        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        if self.args.use_entropy_loss:
            status["entropy_loss"] = entropy_loss.item()
        return status

    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        # critic loss
        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def save_checkpoints(self, args, global_step, client_states={}):
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_" + self.rolename),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        if self.critic is not None:
            self.strategy.save_ckpt(
                self.critic, os.path.join(args.ckpt_path, "_" + self.critic.rolename), tag, args.max_ckpt_num, args.max_ckpt_mem
            )
