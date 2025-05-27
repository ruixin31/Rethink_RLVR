ROOT_DIR=$(pwd)
WANDB_KEY=1
LR=5e-7
KL_COEF=0.00
BACKBONE="Qwen2.5-Math-7B"
BACKBONE_PATH="Qwen/Qwen2.5-Math-7B"
MAX_LENGTH=3072
MODEL_ID="qwen2.5_math_7b"
DATE=$(date +%m%d)

TASK=DeepScaleR_mv_labeled_qwen2.5_math_7b
REWARD="math"

EXPERIMENT="RLVR-${REWARD}"

mkdir -p ${ROOT_DIR}/logs
mkdir -p ${ROOT_DIR}/outputs

MODEL="${TASK}-${BACKBONE}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${MODEL}/${DATE}/${TASK}-${EXPERIMENT}"


EXP="${DATE}-${MODEL_ID}-${TASK}-${EXPERIMENT}-lr${LR}-kl${KL_COEF}"
LOG_FILE="${ROOT_DIR}/logs/${EXP}.log"


python -m ttrl.cli.train_ppo_naive \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --pretrain "${BACKBONE_PATH}" \
   --save_path "${OUTPUT_DIR}/model" \
   --verify_task "${REWARD}" \
   --verify_task_eval "math" \
   --micro_train_batch_size 4 \
   --train_batch_size 128 \
   --num_episodes 200 \
   --actor_scheduler "constant" \
   --lr_warmup_ratio 0 \
   --save_steps 50 \
   --eval_steps 1 \
   --logging_steps 1 \
   --max_samples 400000 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 16 \
   --n_votes_per_prompt 64 \
   --extra_eval_task_fast "test,AIME2025-TTT@8,AIME-TTT@8,AMC-TTT@8,AMC-TTT@1,MATH-TTT@1" \
   --extra_eval_task_fast_supress_orig_eval \
   --eval_temperature_at_k 0.6 \
   --training_mode "rl" \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len ${MAX_LENGTH} \
   --advantage_estimator "group_norm" \
   --use_kl_loss \
   --temperature 1.0 \
   --eval_temperature 0.0 \
   --lambd 1.0 \
   --gamma 1.0 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate ${LR} \
   --critic_learning_rate 9e-6 \
   --init_kl_coef ${KL_COEF} \
   --prompt_data "json@${ROOT_DIR}/data/${TASK}" \
   --input_key "prompt" \
   --label_key "answer" \
   --max_ckpt_num 1 \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --flash_attn \
   --use_wandb ${WANDB_KEY} \
   --wandb_project SpuriousRewardRLVR \
   --wandb_run_name ${EXP} \
   --ckpt_path "${OUTPUT_DIR}/ckpt" 