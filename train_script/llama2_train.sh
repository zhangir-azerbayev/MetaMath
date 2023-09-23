#!/bin/bash
#SBATCH --job-name="llemma_instruct"
# #SBATCH --account=dw87
#SBATCH --comment="eleutherai"
#SBATCH --qos=dw87
#SBATCH --partition=dw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=llama2_7b_metainstruct_%j.out
#SBATCH --error=llama2_7b_metainstruct_%j.out
#SBATCH --time=3-00:00:00

# BYU cluster


module load openmpi cuda/11.8


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


cd /fsx/proj-mathlm/instruct/MetaMath
BASE_DIR=$(pwd)
TRAIN_FILE=/fsx/proj-mathlm/instruct/MetaMathQA/MetaMath-40K.json
MODEL=/fsx/proj-mathlm/downloaded-weights/Llama-2-7b-hf
CONFIG=${BASE_DIR}/deepspeed_config.json

OUTDIR=./model/llama2_metainstruct

NUM_STEPS=938

deepspeed --include localhost:0,1,2,3,4,5,6,7  ${BASE_DIR}/train_math.py \
    --deepspeed ${CONFIG} \
    --model_name_or_path ${MODEL} \
    --data_path ${TRAIN_FILE} \
    --data_length 40000 \
    --bf16 \
    --output_dir ${OUTDIR} \
    --max_steps ${NUM_STEPS} \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps ${NUM_STEPS} \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --logging_dir "$OUTDIR" \
    --report_to="tensorboard" \
