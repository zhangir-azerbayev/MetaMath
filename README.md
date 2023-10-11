# Llemma: MetaMathQA Finetunes

Code for finetuning the Code Llama 7B and Llemma 7B models on the MetaMathQA dataset.

Instructions for replicating the finetuning experiments in Azerbayev et al. (2023) are below.

### Replication Instructions
First, modify `env_setup.sh` to declare the `BASE_DIR` and `TRAIN_FILE` environment variables correctly. Then, from the base directory of this repository, run
```
./train_scipt/train_llama2_full.sh
./train_script/train_codellama_full.sh
./train_script/train_llemma_7b_full.sh
```
Note that the `train_llama2_full.sh` script is designed to replicate the experiments in Yu et al. (2023). The scripts are designed for an 8xA100 80GB configuration: modify them for your hardware as appropriate.

Once the models have finished finetuning, run
```
./eval_scripts/llama2_eval.sh
./eval_scripts/codellama_eval.sh
./eval_scripts/llemma_eval.sh
```
from the base directory of this repository to replicate evaluation results.

### Citation

```
# Add Llemma citation

@misc{yu2023metamath,
      title={MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models}, 
      author={Longhui Yu and Weisen Jiang and Han Shi and Jincheng Yu and Zhengying Liu and Yu Zhang and James T. Kwok and Zhenguo Li and Adrian Weller and Weiyang Liu},
      year={2023},
      eprint={2309.12284},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
