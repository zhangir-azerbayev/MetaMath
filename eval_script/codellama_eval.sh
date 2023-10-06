#!/bin/bash

source env_setup.sh
cd ${BASE_DIR}

MODEL=model/codellama_metainstruct_full/checkpoint-9258 

python eval_gsm8k.py --model $MODEL --data_file data/test/GSM8K_test.jsonl --tensor_parallel_size 1
python eval_math.py --model $MODEL --data_file data/test/MATH_test.jsonl --tensor_parallel_size 1
