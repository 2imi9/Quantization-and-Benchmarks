# Intro

This custom quantization and dequantization toolkit is self-contained.
That is, it is required to perform quantization and dequantization using the toolkit.
It is highly risky to perform dequantization using this toolkit but quantization is performed with others like llm-compressors or compressed tensors.

# DType

## MXFP

The quantization and dequantization are performed with OAI MXFP Triton kernels

## NVFP4

The quantization and dequantization are performed with scripts sourced from torchao, which is a mimic of Nvidia NVFP4 implementation