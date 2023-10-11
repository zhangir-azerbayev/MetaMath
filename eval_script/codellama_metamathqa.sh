#!/bin/bash
#SBATCH --job-name=mathlm
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=12          # Number of cores per tasks
#SBATCH --gres=gpu:1                 # Number of gpus
#SBATCH --output=codellama_7b_metamathqa_eval_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --error=codellama_7b_metamathqa_eval_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --account=neox
# #SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --requeue

module load openmpi cuda/11.8

cd /fsx/proj-mathlm/instruct/MetaMath

source /admin/home-hailey/miniconda3/bin/activate metainstruct

CONDA_HOME=/admin/home-hailey/miniconda3/envs/metainstruct
CUDNN_HOME=/fsx/hailey/cudnn-linux-x86_64-8.6.0.163_cuda11-archive

export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_HOME/include:$CPATH

export PATH=$CONDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_HOME/include:$CPATH

export LD_PRELOAD=/usr/local/cuda-11.7/lib/libnccl.so

export HF_DATASETS_CACHE="/fsx/proj-mathlm/.cache"
export TRANSFORMERS_CACHE="/fsx/proj-mathlm/.cache"

MODEL=zhangirazerbayev/codellama_7b_metamathqa

python eval_gsm8k.py --model $MODEL --data_file data/test/GSM8K_test.jsonl --tensor_parallel_size 1
python eval_math.py --model $MODEL --data_file data/test/MATH_test.jsonl --tensor_parallel_size 1
