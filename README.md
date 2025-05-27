<div align="center">

![thinking-spongebob](figs/thinking-spongebob.png)

# 💭 Spurious Rewards: Rethinking Training Signals in RLVR
  
[Rulin Shao*](https://rulinshao.github.io/), [Shuyue Stella Li*](https://stellalisy.com/), [Rui Xin*](https://ruixin31.github.io/), [Scott Geng*](https://www.scottgeng.com/), [Yiping Wang](https://ypwang61.github.io/), [Sewoong Oh](https://homes.cs.washington.edu/~sewoong/), [Simon Shaolei Du](https://simonshaoleidu.com/), [Nathan Lambert](https://www.natolambert.com/), [Sewon Min](https://www.sewonmin.com/), [Ranjay Krishna](https://www.ranjaykrishna.com/index.html), [Yulia Tsvetkov](https://homes.cs.washington.edu/~yuliats/), [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/), [Pang Wei Koh](https://koh.pw/), [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/luke-zettlemoyer/)
</div>

<div align="center">

[![Github](https://img.shields.io/badge/Github-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/ruixin31/Rethink_RLVR)
[![Website](https://img.shields.io/badge/Site-000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f) 
[![Paper](https://img.shields.io/badge/Paper-000000.svg?style=for-the-badge&logo=arxiv&logoColor=white)](paper/rethink-rlvr.pdf) 
[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/StellaLisy/status/1927392717593526780)

</div>


## Setup

```sh
# Our codebase is based on TTRL (https://github.com/PRIME-RL/TTRL).
git clone git@github.com:ruixin31/Rethink_RLVR
cd code

conda create -n rethink-rlvr python=3.10 
pip install -r requirements.txt
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


## Paper

ArXiv coming soon! In the meantime, here's [the link](paper/rethink-rlvr.pdf) to our paper.

## Citation

```bibtex
@misc{shao2025spurious,
  title={Spurious Rewards: Rethinking Training Signals in RLVR},
  author={Rulin Shao and Shuyue Stella Li and Rui Xin and Scott Geng and Yiping Wang and Sewoong Oh and Simon Shaolei Du and Nathan Lambert and Sewon Min and Ranjay Krishna and Yulia Tsvetkov and Hannaneh Hajishirzi and Pang Wei Koh and Luke Zettlemoyer},
  year={2025},
  howpublished={\url{https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f}},
  note={Notion Blog}
}
```


## Acknowledgments
This repository is built based on [TTRL](https://github.com/PRIME-RL/TTRL), which is built on top of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We added asynchronous evaluation among other custom features to the codebase. 