import itertools
import math
import os
import socket
from tqdm import tqdm
from typing import Callable, Dict, List

import deepspeed
import ray
import torch
from transformers.trainer import get_scheduler
from datasets import Dataset

from ttrl.datasets.prompts_dataset import PromptDatasetWithLabel
from ttrl.datasets.sft_dataset import SFTDataset
from ttrl.models.actor import Actor
from ttrl.trainer.ppo_trainer import PPOTrainer
from ttrl.trainer.experience_maker import Experience, ExperienceMaker
from ttrl.helper.deepspeed import DeepspeedStrategy
from ttrl.helper.distributed_utils import init_process_group

from ttrl.models.ray_launcher import BasePPORole


class ActorPPOTrainer(PPOTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        remote_rm_url: List[str] = None,
        critic_train_remote: bool = False,
        actor_name: str = "ttrl",
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        self.critic_train_remote = critic_train_remote
        self.wait_for_refs = []

        self.experience_maker = ExperienceMaker(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            packing_samples=self.strategy.args.packing_samples,
        )

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
            # https://github.com/OpenRLHF/OpenRLHF/issues/313
            import vllm

            if not vllm.__version__ == "0.4.2" and not vllm.__version__ >= "0.6.4":
                backend = "gloo"
                print(
                    "Warning: using --vllm_sync_backend=gloo for `not vLLM version == 0.4.2 and not vllm.__version__ >= 0.6.4`"
                )

            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    actor_name,
                    backend=backend,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=actor_name,
            )

            ray.get(refs)

        torch.distributed.barrier()

    def ppo_train(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()

        # 2. triger remote critic model training
        if self.critic_train_remote:
            critic_status_ref = self.critic.fit.remote()

        # 3. actor model training
        if global_steps > self.freezing_actor_steps:
            if self.args.training_mode == "sft":
                status = self.sft_train(global_steps)
            else:
                status = super().ppo_train(global_steps)

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                if self.wait_for_refs:
                    print("=======waiting for refs=======")
                    ray.get(self.wait_for_refs)
                    self.wait_for_refs = []
                    print("=======done waiting for refs=======")
                torch.distributed.barrier()
                self._broadcast_to_vllm()
        else:
            status = {}

        # 5. wait remote critic model training done
        if self.critic_train_remote:
            status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        return status

    def fit(self, steps, samples_ref, wait_for_refs=None):
        if wait_for_refs:
            self.wait_for_refs = wait_for_refs
        return super().fit(steps, samples_ref)

    def sft_train(self, global_steps):
        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                self.pretrain_dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for data in pbar:
                status = self.training_step_actor_supervised(data)
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

        # TODO: refer to https://github.com/huggingface/trl/issues/2840#issuecomment-2662747485
        for param in self.actor.model.parameters():
            param.ds_active_sub_modules.clear()
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        return self.training_step_actor(experience)

    def training_step_actor_supervised(self, data) -> Dict[str, float]:
        self.actor.train()
        # data = next(self.pretrain_dataloader)
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
        total_loss = ptx_loss + aux_loss * self.args.aux_loss_coef
        
        self.strategy.backward(total_loss, self.actor, self.actor_optim)
        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        status = {
            "sft_loss": total_loss.item(),
            "actor_lr": self.actor_scheduler.get_last_lr()[0]
        }
        return status

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        # avoid OOM
        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                if torch.distributed.get_rank() == 0:
                    torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                    ray.get(refs)

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.distributed.barrier()

    def _save_checkpoint(self, args, tag, client_states):
        # call remote critic
        if self.critic_train_remote:
            ref = self.critic.save_checkpoint.remote(tag)
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_" + self.rolename),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        # wait
        if self.critic_train_remote:
            ray.get(ref)


@ray.remote(num_gpus=1)
class ActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, 
        strategy: DeepspeedStrategy,
        pretrain, 
        max_steps,
        rolename="actor"):
        args = strategy.args
        self._setup_distributed(strategy)
        self.rolename = rolename
        
        actor = Actor(
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

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None
            
        # self.gt_label_path = os.environ.get("GT_LABEL_PATH", None)
        # # load gt_labels from the pkl file defined in gt_label_path
        # if args.gt_label_path:
        #     if os.path.exists(args.gt_label_path):
        #         # load pkl file
        #         import pickle
        #         with open(args.gt_label_path, "rb") as f:
        #             self.gt_labels = pickle.load(f)
        #     else:
        #         raise FileNotFoundError(f"gt_labels file {args.gt_label_path} not found.")

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        actor_scheduler = get_scheduler(
            getattr(args, "actor_scheduler", "cosine_with_min_lr"),
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        ckpt_path = os.path.join(args.ckpt_path, "_" + self.rolename)
        if args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            # pass

    def init_trainer(self, 
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        critic_train_remote: bool = False):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args
        # configure Trainer
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            actor_name=self.rolename,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            remote_rm_url=remote_rm_url,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            entropy_coef=args.entropy_coef,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            rolename=self.rolename
        )

        # broadcast checkpoint
        ckpt_path = os.path.join(args.ckpt_path, "_" + self.rolename)
        if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
            torch.distributed.barrier()
            self.trainer._broadcast_to_vllm()

    def create_pretrain_dataloader(self, pretrain_dataset: Dataset):
        pretrain_dataloader = self.strategy.setup_dataloader(
                    pretrain_dataset,
                    self.strategy.args.micro_train_batch_size,
                    True,
                    True,
                    pretrain_dataset.collate_fn,
        )
        if self.strategy.args.training_mode in ["both", "mix", "rl"]:
            pretrain_dataloader = itertools.cycle(
                iter(
                    pretrain_dataloader
                )
            )
        return pretrain_dataloader

    def fit(self, steps, samples_ref, pretrain_dataset=None, wait_for_refs=None):
        if pretrain_dataset is not None:
            self.trainer.pretrain_dataloader = self.create_pretrain_dataloader(pretrain_dataset)
        status = self.trainer.fit(steps, samples_ref, wait_for_refs)
        self.trainer.pretrain_dataloader = None
        status["agent"] = self.rolename
        return status

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            None,
            args.save_path,
        )
