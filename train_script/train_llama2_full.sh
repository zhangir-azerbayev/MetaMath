#!/bin/bash
source env_setup.sh
cd ${BASE_DIR}

MODEL=meta-llama/Llama-2-7b-hf
CONFIG=${BASE_DIR}/deepspeed_config.json

OUTDIR=${BASE_DIR}/model/llama2_metainstruct_full

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
