#!/bin/bash

# Loop over datasets "mmlu-redux, math500, livecode_v5 for measurement"
datasets=(
    "ifeval"
    "math500"
    "mmlu-redux"
    "livecode_v5"
    "aime25"
)

# Base output path
BASE_OUTPUT_PATH="/app/outputs"

OUTPUT_PREFIX="qwen3_30b_a3b_instruct_int4_w4a16_g32"
MODEL_NAME="/app/outputs/quantized_ckpts/Qwen3-30B-A3B-Instruct-2507-GPTQ-W4A16-G32"

# Loop through all ruler datasets
for dataset in "${datasets[@]}"; do
    echo "Running inference for dataset: $dataset for model $MODEL_NAME"

    # Create output directory path
    OUTPUT_DIR="${BASE_OUTPUT_PATH}/${OUTPUT_PREFIX}_${dataset}_2"
    echo $OUTPUT_DIR

    # Run the inference
    python3 ./infer.py --config-name=qwen3_30B_A3B_Instruct_bf16 \
        eval_dataset="$dataset" \
        output_dir="$OUTPUT_DIR" \
        eval_predictor=vllm \
        predictor_conf.vllm.tensor_parallel_size=4 \
        model_name=${MODEL_NAME}

    echo "Completed dataset: $dataset"
    echo "-----------------------------------"
done
