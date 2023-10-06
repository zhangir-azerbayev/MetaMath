#!/bin/bash
#SBATCH --job-name="llemma_metainstruct"
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
#SBATCH --output=train_llemma_7b_full_%j.out
#SBATCH --error=train_llemma_7b_full_%j.out
#SBATCH --time=3-00:00:00

# BYU cluster


source /home/hailey81/miniconda3/bin/activate metainstruct

which python

export LD_LIBRARY_PATH=/home/hailey81/miniconda3/envs/metainstruct/lib/
export PATH=/home/hailey81/cuda_install/bin:$PATH

ln -s /home/hailey81/miniconda3/envs/metainstruct-updated/bin/gcc/ ~/.local/bin/gcc
export PATH=$HOME/.local/bin:$PATH

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /home/za2514/compute/instruct/MetaMath
BASE_DIR=$(pwd)
TRAIN_FILE=/nobackup/scratch/usr/za2514/instruct/MetaMathQA/MetaMathQA-395K.json
MODEL=open-web-math/codellama_7b_200btok_step42000
CONFIG=${BASE_DIR}/deepspeed_config.json

OUTDIR=./model/llemma_7b_metainstruct_full

NUM_STEPS=9258

deepspeed --include localhost:0,1,2,3,4,5,6,7  ${BASE_DIR}/train_math.py \
    --deepspeed ${CONFIG} \
    --model_name_or_path ${MODEL} \
    --data_path ${TRAIN_FILE} \
    --data_length 395000 \
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
