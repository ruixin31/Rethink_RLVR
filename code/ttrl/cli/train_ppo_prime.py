import argparse
from datetime import datetime

from ttrl.helper.utils import get_strategy

from ttrl.controller.prime_controller import PrimeController

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

    assert (
        actor_world_size & (actor_world_size - 1)
    ) == 0, f"actor_world_size must be power of 2, got {actor_world_size}"

    if args.critic_pretrain:
        critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
        assert (
            critic_world_size & (critic_world_size - 1)
        ) == 0, f"critic_world_size must be power of 2, got {critic_world_size}"
        assert (
            actor_world_size % critic_world_size == 0
        ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"

    assert args.zero_stage != 3 or args.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"


def train(args):
    _validate_args(args)
    # configure strategy
    strategy = get_strategy(args)

    # start controller
    controller = PrimeController(strategy=strategy)
    controller.build_env()
    controller.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1,
                        help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int,
                        default=8, help="number of gpus per node for reference")

    parser.add_argument("--reward_num_nodes", type=int,
                        default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int,
                        default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int,
                        default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int,
                        default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int,
                        default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )
    
    # prime reward model
    parser.add_argument("--prime_ref_num_nodes", type=int, default=1,
                        help="number of nodes for reference")
    parser.add_argument("--prime_ref_num_gpus_per_node", type=int,
                        default=8, help="number of gpus per node for reference")
    parser.add_argument("--prime_num_nodes", type=int,
                        default=1, help="number of nodes for prime")
    parser.add_argument("--prime_num_gpus_per_node", type=int,
                        default=8, help="number of gpus per node for prime")
    parser.add_argument(
        "--colocate_prime_ref",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--prime_beta", type=float, default=0.05, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--prime_granularity", type=str, default="token", help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument("--prime_nll_loss",
                        action="store_true", default=False)
    parser.add_argument("--prime_nll_loss_coef", type=float,
                        default=0, help="nll balancing loss")
    parser.add_argument("--prime_grad_clip", type=float,
                        default=10.0, help="max grad norm")
    parser.add_argument("--prime_adam_betas", type=float, nargs=2,
                        default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--prime_l2", type=float, default=0.0,
                        help="weight decay loss")
    parser.add_argument("--prime_score_coef", type=float, default=5.0,
                        help="weight of prime score")
    parser.add_argument("--verifier_score_coef", type=float, default=1.0,
                        help="weight of verifier score")
    parser.add_argument("--prime_learning_rate", type=float, default=1e-6)
    parser.add_argument("--prime_pretrain", type=str, default=None,
                        help="HF model name or path")
    parser.add_argument("--prime_scheduler", type=str, default="constant_with_warmup",
                        help="HF model name or path")
    parser.add_argument("--prime_lr_warmup_ratio", type=float, default=0.0)
    parser.add_argument("--batch_norm",
                        action="store_true", default=False)
    
    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str,
                        default="gloo", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--enable_prefix_caching",
                        action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true",
                        default=False, help="Disable CUDA graph in vLLM")

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str,
                        default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint",
                        action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-
                        1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2,
                        help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing",
                        action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true",
                        default=False, help="Enable bfloat16")
    # Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true",
                        help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1,
                        help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true",
                        default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu",
                        action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true",
                        default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str,
                        default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache",
                        action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant",
                        action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer",
                        action="store_true", default=False)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples",
                        action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str,
                        nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int,
                        default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int,
                        default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None,
                        help="deprecated max_len")
    parser.add_argument("--max_samples", type=int,
                        default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float,
                        default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0,
                        help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float,
                        default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float,
                        default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float,
                        default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float,
                        default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int,
                        default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int,
                        default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true",
                        default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freezing_actor_steps", type=int,
                        default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true",
                        default=False, help="Save critic model")
    parser.add_argument("--use_kl_loss", action="store_true",
                        default=False, help="Use kl loss")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float,
                        default=0.01, help="KL penalty in PPO")
    parser.add_argument(
        "--use_kl_estimator_k3",
        action="store_true",
        default=False,
        help=(
            "Use the k3 estimator in http://joschu.net/blog/kl-approx.html"
            "to ensure the KL divergence calculated is non-negative"
        ),
    )
    parser.add_argument("--aux_loss_coef", type=float,
                        default=0, help="MoE balancing loss")
    parser.add_argument("--adam_betas", type=float, nargs=2,
                        default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float,
                        nargs=2, default=(-10, 10), help="Reward clip range")

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "grpo", "reinforce_baseline", "group_norm"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo",
    )

    parser.add_argument("--training_mode", type=str,
                        default="rl", help="training mode, rl / sft / both /mix")

    #  Models
    parser.add_argument("--pretrain", type=str, default=None,
                        help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str,
                        default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str,
                        default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str,
                        default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload",
                        action="store_true", default=False)

    parser.add_argument("--verify_task", type=str, default="math", choices=[
                        "math", "simplerl", "review", "review_group", "think"], help="which verify to used for the task")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str,
                        default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str,
                        default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")

    parser.add_argument("--input_key", type=str,
                        default="input", help="JSON dataset key")
    parser.add_argument("--label_key", type=str,
                        default="label", help="JSON dataset key")
    parser.add_argument("--output_key", type=str,
                        default="output", help="JSON dataset key")
    parser.add_argument("--add_think_token", type=int,
                        default=0, help="add <think> token after the prompt for r1")
    parser.add_argument("--add_prompt_suffix", type=str,
                        default=None, help="add suffix after the prompt")

    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str,
                        default="ttrl_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str,
                        default=None, help="TensorBoard logging path")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    args = parser.parse_args()

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        if args.reward_pretrain is not None:
            args.critic_pretrain = args.reward_pretrain.split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.advantage_estimator == "rloo":
        assert args.n_samples_per_prompt > 1, "RLOO requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.vllm_num_engines >= 1 and args.enable_prefix_caching:
        import vllm
        if vllm.__version__ < "0.7.0":
            args.enable_prefix_caching = False
            print("[Warning] Disable prefix cache because vLLM updates weights without updating the old KV Cache for vLLM version below 0.7.0.")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples:
        if not args.flash_attn:
            print(
                "[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            args.flash_attn = True
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."
        assert not args.pretrain_data, "`--pretrain_data` is not supported with `--packing_samples` yet."

    train(args)
