import re
import json
import shutil
from pathlib import Path
from typing import Optional

import hydra
import torch
import kernels
from loguru import logger
from omegaconf import DictConfig
from safetensors.torch import load_file, save_file

from nvfp4_utils import nvfp4_quantize, per_tensor_amax_to_scale


triton_kernels_hub = kernels.get_kernel("kernels-community/triton_kernels")


def quantize_to_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to 4-bit MXFP format.
    """
    assert weight.dtype == torch.bfloat16
    assert weight.ndim == 2

    # Get the kernel
    downcast_to_mxfp = triton_kernels_hub.numerics_details.mxfp.downcast_to_mxfp
    w, w_scale = downcast_to_mxfp(weight, torch.uint8, axis=1)
    assert w.shape == (weight.shape[0], weight.shape[1] // 2)
    assert w_scale.shape == (weight.shape[0], weight.shape[1] // 32)
    return w, w_scale


def quantize_to_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to 8-bit MXFP format.
    """
    assert weight.dtype == torch.bfloat16
    assert weight.ndim == 2

    # Get the kernel
    downcast_to_mxfp = triton_kernels_hub.numerics_details.mxfp.downcast_to_mxfp
    w, w_scale = downcast_to_mxfp(weight, torch.float8_e4m3fn, axis=1)
    assert w.shape == weight.shape
    assert w_scale.shape == (weight.shape[0], w.shape[1] // 32)
    return w, w_scale


def quantize_to_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to 4-bit NVFP4 format.
    """
    assert weight.dtype == torch.bfloat16
    assert weight.ndim == 2

    per_tensor_scale = per_tensor_amax_to_scale(torch.max(torch.abs(weight)).to(torch.float32))
    w_scale, w = nvfp4_quantize(weight, per_tensor_scale=per_tensor_scale)
    assert w.shape == (weight.shape[0], weight.shape[1] // 2)
    assert w_scale.shape == (weight.shape[0], weight.shape[1] // 16)
    return w, w_scale, per_tensor_scale


@hydra.main(config_path="conf", config_name="quant", version_base=None)
def main(cfg: DictConfig):
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir) if cfg.output_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)

    tgt_patterns = [re.compile(p) for p in cfg.patterns]

    # read all safe tensors in a directory and quantize them one by one
    for file in Path(input_dir).glob("*.safetensors"):
        logger.info(f"Loading {file}")
        model = load_file(file)
        logger.info(f"Model keys: {model.keys()}")

        tgts = [k for k in model.keys() if any(p.match(k) for p in tgt_patterns)]
        logger.info(f"Found {len(tgts)} target tensors")

        for tgt in tgts:
            logger.info(f"== Quantizing {tgt} into {cfg.quant_type} ==")
            weight = model[tgt]
            logger.info(f"Weight: {weight.shape}, {weight.dtype}")

            w_global_scale = None
            if cfg.quant_type == "mxfp4":
                w, w_scale = quantize_to_mxfp4(weight.to("cuda:0"))
            elif cfg.quant_type == "mxfp8":
                w, w_scale = quantize_to_mxfp8(weight.to("cuda:0"))
            elif cfg.quant_type == "nvfp4":
                w, w_scale, w_global_scale = quantize_to_nvfp4(weight.to("cuda:0"))
                logger.info(f"Global scale: {w_global_scale}")
            else:
                raise ValueError(f"Invalid quant type: {cfg.quant_type}")

            del model[tgt]
            model[tgt.replace(".weight", ".weight_packed")] = w
            model[tgt.replace(".weight", ".weight_scale")] = w_scale
            if w_global_scale is not None:
                model[tgt.replace(".weight", ".weight_global_scale")] = w_global_scale

        if output_dir is not None:
            output_path = output_dir / file.name
            logger.info(f"Saving to {output_path}")
            save_file(model, output_path)

    # move other files from input dir to output dir
    for file in input_dir.glob("*"):
        if file.is_file() and all(not file.name.endswith(x) for x in [".safetensors", ".pt"]):
            shutil.copy(file, output_dir / file.name)

    # read config file and remove quantization related configs
    logger.info(f"Add quantization configs")
    quantization_config = {
        "modules_to_convert": list(cfg.patterns),  # Convert ListConfig to regular list
        "quant_method": cfg.quant_type
    }
    model_cfg_path = output_dir / "config.json"
    with open(model_cfg_path, "r") as f:
        model_cfg = json.load(f)
    model_cfg["quantization_config"] = quantization_config
    with open(model_cfg_path, "w") as f:
        json.dump(model_cfg, f)
    logger.info(model_cfg)


if __name__ == "__main__":
    main()
