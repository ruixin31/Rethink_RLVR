<div align="center">

![thinking-spongebob](figs/thinking-spongebob.png)

# 💭 Spurious Rewards: Rethinking Training Signals in RLVR
  
[Rulin Shao*](https://rulinshao.github.io/), [Shuyue Stella Li*](https://stellalisy.com/), [Rui Xin*](https://ruixin31.github.io/), [Scott Geng*](https://www.scottgeng.com/), [Yiping Wang](https://ypwang61.github.io/), [Sewoong Oh](https://homes.cs.washington.edu/~sewoong/), [Simon Shaolei Du](https://simonshaoleidu.com/), [Nathan Lambert](https://www.natolambert.com/), [Sewon Min](https://www.sewonmin.com/), [Ranjay Krishna](https://www.ranjaykrishna.com/index.html), [Yulia Tsvetkov](https://homes.cs.washington.edu/~yuliats/), [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/), [Pang Wei Koh](https://koh.pw/), [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/luke-zettlemoyer/)
</div>

<div align="center">

[![Github](https://img.shields.io/badge/Github-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/ruixin31/Rethink_RLVR)
[![Website](https://img.shields.io/badge/Site-000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f) 
[![Paper](https://img.shields.io/badge/Paper-000000.svg?style=for-the-badge&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2506.10947) 
[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/StellaLisy/status/1927392717593526780)
[![Wandb](https://img.shields.io/badge/📁_reproduction_W&B-000000?style=for-the-badge&logo=wandb&logoColor=white)](https://wandb.ai/rx31/SpuriousRewardRLVR)
[![Models](https://img.shields.io/badge/Models-000000?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/stellalisy/spurious-rewards-684a38b8eeb32273c287a4db)

</div>


## Setup

```sh
# Our codebase is based on TTRL (https://github.com/PRIME-RL/TTRL).
git clone git@github.com:ruixin31/Spurious_Rewards
cd code

conda create -n spurious-rewards python=3.10 
conda activate spurious-rewards

pip install -r requirements.txt
pip install flash_attn==2.7.0.post2
pip install -e .
```

## Training
```sh
bash scripts/rlvr_deepscaler_grpo_qwen_ground_truth.sh
```

## Configurations

### Data
We include filtered and majority-labeled data in the paper. You may find a complete list in the `code/data` directory. For example, the ground truth data is termed `DeepScaleR`, and Llama 3.2 3B instruct labeled data, filtered to keep only the incorrect labels, is in the `DeepScaleR_mv_labeled_llama3.2_3b_instruct_incorrect` folder. You may change the data source by changing the variable `TASK` in `code/scripts/rlvr_deepscaler_grpo_qwen_ground_truth.sh`. 

### Rewards
We include a list of rewards used in the paper below. Furthermore, note that for models without a chat template, be sure to add `_r1_only` as the suffix. You may change the reward function by changing the variable `REWARD` in `code/scripts/rlvr_deepscaler_grpo_qwen_ground_truth.sh`. 

- `math`: Mathematical equivalence reward, which is the default
- `box_only_format`: Box-only formatting reward
- `contain_python_wo_backticks`: Mentioning of Python reward
- `random0.5`: Random reward with 50% returning 1


## Evaluations
To reproduce our evaluation results, use the following commands:

```sh
cd code

# For MATH-500 evaluation (requires NVIDIA A100 80GB PCIe for exact reproduction)
python scripts/eval_checkpoint.py --model_path Qwen/Qwen2.5-Math-7B --datasets MATH-500,AIME-2024,AIME-2025,AMC

# For MATH-500 evaluation matching our reported scores in wandb using checkpoints (requires NVIDIA H200 for exact reproduction)
python scripts/eval_checkpoint.py --model_path {} --datasets MATH-500,AIME-2024,AIME-2025,AMC --shards 2
```

Note: To exactly reproduce `temperature = 0` results, both the GPU type and `--shards` parameter must match the original evaluation setup. This is because the batch size passed into VLLM can cause generation fluctuations.

## Paper

Here's [the link](http://arxiv.org/abs/2506.10947) to our paper.

## Citation

```bibtex
@misc{shao2025spuriousrewardsrethinkingtraining,
      title={Spurious Rewards: Rethinking Training Signals in RLVR}, 
      author={Rulin Shao and Shuyue Stella Li and Rui Xin and Scott Geng and Yiping Wang and Sewoong Oh and Simon Shaolei Du and Nathan Lambert and Sewon Min and Ranjay Krishna and Yulia Tsvetkov and Hannaneh Hajishirzi and Pang Wei Koh and Luke Zettlemoyer},
      year={2025},
      eprint={2506.10947},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.10947}, 
}
```


## Acknowledgments
This repository is built based on [TTRL](https://github.com/PRIME-RL/TTRL), which is built on top of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We added asynchronous evaluation among other custom features to the codebase. 