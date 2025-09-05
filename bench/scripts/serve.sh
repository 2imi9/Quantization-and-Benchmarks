#!/bin/bash

BASE_OUTPUT_PATH="/app/outputs"
OUTPUT_PREFIX="qwen3_30b_a3b_instruct_bf16"
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"

OUTPUT_DIR="${BASE_OUTPUT_PATH}/${OUTPUT_PREFIX}_vllm_serve"

python3 ./vllm_serve.py --config-name=qwen3_30B_A3B_Instruct_bf16 \
    eval_dataset="$dataset" \
    output_dir="$OUTPUT_DIR" \
    eval_predictor=vllm \
    predictor_conf.vllm.tensor_parallel_size=8 \
    predictor_conf.vllm.enforce_eager=true \
    predictor_conf.vllm.max_seq_len=32000 \
    predictor_conf.vllm.max_num_batched_tokens=32000 \
    predictor_conf.vllm.max_num_seqs=1 \
    predictor_conf.vllm.gpu_memory_utilization=0.90 \
    predictor_conf.vllm.cpu_offload_gb=30 \
    model_name=${MODEL_NAME}
