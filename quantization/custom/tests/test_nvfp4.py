import re
import torch
import pytest
from loguru import logger
from safetensors.torch import load_file


@pytest.mark.nvfp4
@pytest.mark.parametrize("nvfp4_dequantized_path, bf16_path", [
    ("/app/outputs/quantized_ckpts/Qwen3-1.7B-nvfp4a16-torchao-dequantized-BF16/model-00001-of-00002.safetensors",
     "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/model-00001-of-00002.safetensors")
])
def test_nvfp4_error(nvfp4_dequantized_path: str, bf16_path: str):
    # Measure the error between dequantize nvfp4 and bf16
    nvfp4_dequantized_model = load_file(nvfp4_dequantized_path)
    bf16_model = load_file(bf16_path)

    patterns = [re.compile(p) for p in [
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
        "model.layers.*.self_attn.o_proj.weight",
        "model.layers.*.mlp.up_proj.weight",
        "model.layers.*.mlp.down_proj.weight",
        "model.layers.*.mlp.gate_proj.weight"
    ]]

    keys_to_compare = [k for k in nvfp4_dequantized_model if any(p.match(k) for p in patterns)]
    logger.info(f"# Keys to compare: {len(keys_to_compare)}")

    for key in keys_to_compare:
        logger.info(f"== Comparing {key} ==")
        nvfp4_dequantized_weight = nvfp4_dequantized_model[key]
        bf16_weight = bf16_model[key]
        
        nd_max = torch.max(torch.abs(nvfp4_dequantized_weight))
        bf16_max = torch.max(torch.abs(bf16_weight))
        logger.info(f"nd_max: {nd_max}, bf16_max: {bf16_max}")

        nd_min = torch.min(nvfp4_dequantized_weight)
        bf16_min = torch.min(bf16_weight)
        logger.info(f"nd_min: {nd_min}, bf16_min: {bf16_min}")

        nd_mean = torch.mean(nvfp4_dequantized_weight)
        bf16_mean = torch.mean(bf16_weight)
        logger.info(f"nd_mean: {nd_mean}, bf16_mean: {bf16_mean}")

        nd_std = torch.std(nvfp4_dequantized_weight)
        bf16_std = torch.std(bf16_weight)
        logger.info(f"nd_std: {nd_std}, bf16_std: {bf16_std}")

        mse = torch.mean((nvfp4_dequantized_weight - bf16_weight) ** 2)
        logger.info(f"mse: {mse}")

        rmse = torch.sqrt(mse)
        logger.info(f"rmse: {rmse}")
