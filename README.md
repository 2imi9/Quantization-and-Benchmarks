# Benchmark Low-Bit LLMs: GPT-OSS & Qwen3

A comprehensive benchmarking framework for evaluating quantized GPT-OSS and Qwen3 models across multiple datasets and precision levels.

## Quick Start

### Setup Environment
```bash
cd ./dockers
sudo ./build_docker.sh ./[dockerfile_name]
sudo ./launch_docker.sh [container_name]
```

### Run Evaluation
```bash
cd bench
python3 ./infer.py eval_dataset=mmlu-redux output_dir=/app/outputs/qwen3_4B_bf16_mmlu_redux
python3 ./compute_metric.py -c /app/outputs/qwen3_4B_bf16_mmlu_redux
```

## Supported Models

### GPT-OSS
- **GPT-OSS-20B**: Original MXFP4 quantization format

### Qwen3 Model Family
- **Qwen3-1.7B**: Base model with comprehensive quantization support
  - Precision levels: BF16, INT8, FP8, NVFP4, MXFP4, INT4
  - Thinking mode variants available
- **Qwen3-4B**: Mid-size model with strong performance
  - Precision levels: BF16, NVFP4
  - Thinking/non-thinking mode support
- **Qwen3-30B-A3B-Instruct**: Large instruction-tuned model
  - Precision levels: BF16, NVFP4, MXFP4, INT8-W8A8
  - Optimized for instruction following and reasoning tasks

## Datasets

### Core Benchmarks
- **MMLU-Redux** (29 subjects): Multi-task language understanding
- **Math-500**: Mathematical reasoning with LaTeX parsing
- **LiveCodeBench V5**: Code generation (2024-10 to 2025-02)
- **IFEval**: Instruction following evaluation
- **AIME 2025**: Competition-level mathematics

### Long Context (RULER-32K)
- **NIAH**: Needle in haystack variants
- **CWE/FWE**: Word extraction tasks  
- **VT**: Variable tracking

## Quantization Methods

### GPTQ (via llm-compressor)
```bash
# INT4 Group-32 quantization
python3 quantization/qwen3/gptq_int4_group32.py

# W4A8 quantization  
python3 quantization/qwen3/gptq_int4_w4a8.py
```

### Custom Formats
```bash
cd quantization/custom

# NVFP4 quantization
python3 to_mxfp.py --config-name=quant quant_type=nvfp4

# Dequantize to BF16
python3 to_bf16.py --config-name=dequant quant_type=nvfp4
```

## Configuration

### Model Configs (`bench/conf/`)
- `qwen3_4B_bf16.yaml`: Base Qwen3-4B configuration
- `qwen3_30B_A3B_Instruct_bf16.yaml`: Instruction model
- `gptoss_20B.yaml`: GPT-OSS configuration

### Dataset Parameters
Each dataset supports:
- Temperature, top_p, top_k sampling
- Sequence length (4K-32K)
- Thinking mode toggle
- Multi-sequence generation

## Evaluation Pipeline

### Single Evaluation
```bash
python3 ./infer.py \
    eval_dataset=mmlu-redux \
    output_dir=/app/outputs/qwen3_4B_bf16_mmlu_redux \
    eval_predictor=vllm
```

### Batch Evaluation
```bash
# Multiple datasets
bash bench/scripts/custom.sh

# Long context evaluation
bash bench/scripts/ruler.sh
```

### Model Comparison
```bash
# Compare quantization levels
for precision in bf16 int4 int8; do
    python3 ./infer.py \
        model_config=qwen3_4B_${precision}.yaml \
        eval_dataset=mmlu-redux \
        output_dir=/app/outputs/qwen3_4B_${precision}_mmlu_redux
done
```

## Results

### Current Benchmarks

| Model | Precision | MMLU-Redux | Reference |
|-------|-----------|------------|-----------|
| Qwen3-4B | BF16 | 72.2 | 77.3 |

### Performance Targets
- **MMLU-Redux**: 70%+ accuracy across quantization levels
- **Math-500**: 90%+ with reasoning tokens
- **LiveCode V5**: 60%+ Pass@1 
- **RULER-32K**: 80%+ retrieval accuracy

## Advanced Features

### vLLM Integration
- Pipeline/tensor parallelism for multi-GPU
- Dynamic batching with memory optimization
- CPU offloading for large models

### Custom Quantization
- NVFP4: 4-bit with E4M3 scales
- MXFP: Microscaling formats (4-bit/8-bit)
- Block-wise quantization with triton kernels

### Evaluation Metrics
- **Pass@k**: Code generation success rates
- **Math-verify**: LaTeX answer verification
- **String matching**: Long context retrieval
- **Instruction following**: Strict/loose compliance

## File Structure

```
├── bench/                  # Core framework
│   ├── conf/              # Model configurations  
│   ├── dataset/           # Dataset implementations
│   ├── predictor/         # Inference backends
│   └── scripts/          # Batch evaluation
├── quantization/          # Quantization toolkit
│   ├── qwen3/            # GPTQ scripts
│   └── custom/           # NVFP4/MXFP methods
├── dockers/              # Container configs
└── outputs/              # Results storage
```

## Development

### Adding Models
1. Create config in `bench/conf/model_name.yaml`
2. Add chat template in `bench/chat_template/model_name.py`
3. Update predictor for custom quantization

### Adding Datasets
1. Implement `BaseDataset` in `bench/dataset/dataset_name.py`
2. Register in `bench/dataset/__init__.py`
3. Add evaluation metrics

### Testing
```bash
# Run with debug mode (uses smaller dataset)
python3 ./infer.py eval_dataset=mmlu-redux debug=true

# Test custom quantization
cd quantization/custom
pytest tests/test_nvfp4.py -v

# Test with different container
sudo ./launch_docker.sh test_container bench-gpu:latest
```

## Hardware Requirements

### Minimum
- **GPU**: RTX 3080 (10GB VRAM)
- **RAM**: 32GB system memory
- **Storage**: 100GB for models/results

### Recommended  
- **GPU**: RTX 5090 (32GB VRAM) or 4x RTX 3080
- **RAM**: 128GB system memory
- **Storage**: 500GB NVMe SSD

### Legacy Support
- **V100**: CUDA 7.0 compatibility mode
- **Multi-GPU**: Pipeline parallelism for 30B+ models

## Troubleshooting

### Memory Issues
```bash
# Reduce memory usage
predictor_conf.vllm.gpu_memory_utilization=0.8
predictor_conf.vllm.cpu_offload_gb=20

# Enable tensor parallelism
predictor_conf.vllm.tensor_parallel_size=4
```

### Performance Optimization
```bash
# Dynamic batching
predictor_conf.vllm.max_num_batched_tokens=16384
predictor_conf.vllm.max_num_seqs=8

# Disable features for speed
predictor_conf.vllm.enable_prefix_caching=false
predictor_conf.vllm.enforce_eager=true
```

## Contributing

1. Follow existing code patterns and naming
2. Add tests for new quantization methods  
3. Validate against reference implementations
4. Update documentation and benchmarks

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
