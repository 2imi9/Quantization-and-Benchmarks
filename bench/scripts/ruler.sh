#!/bin/bash

# Define array of dataset names
# "ruler-niah-single-1-32k"
# "ruler-niah-single-2-32k"
# "ruler-niah-single-3-32k"
# "ruler-niah-multikey-1-32k"
# "ruler-niah-multikey-2-32k"
# "ruler-niah-single-3-32k"
# "ruler-niah-multiquery-32k"
# "ruler-niah-multivalue-32k"

datasets=(
    "ruler-niah-multikey-3-32k"
    "ruler-cwe-32k"
    "ruler-vt-32k"
    "ruler-fwe-32k"
)

# Base output path
BASE_OUTPUT_PATH="/app/outputs"

OUTPUT_PREFIX="qwen3_30b_a3b_instruct_int4_w4a16_g32"
MODEL_NAME="/app/outputs/quantized_ckpts/Qwen3-30B-A3B-Instruct-2507-GPTQ-W4A16-G32"

# Loop through all ruler datasets
for dataset in "${datasets[@]}"; do
    echo "Running inference for dataset: $dataset"

    # Create output directory path
    OUTPUT_DIR="${BASE_OUTPUT_PATH}/${OUTPUT_PREFIX}_${dataset}-non-thinking-vllm"
    echo $OUTPUT_DIR

    # Run the inference
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 ./infer.py --config-name=qwen3_30B_A3B_Instruct_bf16 \
        eval_dataset="$dataset" \
        eval_dataset_config=ruler-32k \
        output_dir="$OUTPUT_DIR" \
        eval_predictor=vllm \
        predictor_conf.vllm.cpu_offload_gb=20 \
        predictor_conf.vllm.max_seq_len=32768 \
        predictor_conf.vllm.max_num_batched_tokens=32768 \
        predictor_conf.vllm.max_num_seqs=1 \
        predictor_conf.vllm.gpu_memory_utilization=0.95 \
        predictor_conf.vllm.tensor_parallel_size=4 \
        model_name=${MODEL_NAME}

    echo "Completed dataset: $dataset"
    echo "-----------------------------------"
done

echo "All ruler datasets processed successfully!"
