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

from nvfp4_utils import nvfp4_dequantize

triton_kernels_hub = kernels.get_kernel("kernels-community/triton_kernels")


def dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor
) -> torch.Tensor:
    return nvfp4_dequantize(weight_packed, weight_scale, weight_global_scale)


def dequantize_mxfp(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor
) -> torch.Tensor:
    upcast_from_mxfp = triton_kernels_hub.numerics_details.mxfp.upcast_from_mxfp
    dequant_weight = upcast_from_mxfp(weight_packed.to("cuda:0"), weight_scale.to("cuda:0"), dtype=torch.bfloat16, axis=1)
    return dequant_weight


@hydra.main(config_path="conf", config_name="dequant", version_base=None)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir) if cfg.output_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)

    # read all safe tensors in a directory and dequantize them one by one
    for file in Path(input_dir).glob("*.safetensors"):
        logger.info(f"Loading {file}")
        model = load_file(file)
        logger.info(f"Model keys: {model.keys()}")

        tgts = [k for k in model.keys() if k.endswith(".weight_packed")]
        logger.info(f"Found {len(tgts)} target tensors")

        for tgt in tgts:
            logger.info(f"== Dequantizing {tgt} ==")
            weight_packed = model[tgt]
            weight_scale = model[tgt.replace(".weight_packed", ".weight_scale")]
            logger.info(f"Weight packed: {weight_packed.shape}, {weight_packed.dtype}")
            logger.info(f"Weight scale: {weight_scale.shape}, {weight_scale.dtype}")
            if cfg.quant_type == "nvfp4":
                weight_global_scale = model[tgt.replace(".weight_packed", ".weight_global_scale")]
                logger.info(f"Weight global scale: {weight_global_scale}, {weight_global_scale.dtype}")
                dequant_weight = dequantize_nvfp4(weight_packed, weight_scale, weight_global_scale)
            elif cfg.quant_type == "mxfp4" or cfg.quant_type == "mxfp8":
                weight_global_scale = None
                dequant_weight = dequantize_mxfp(weight_packed, weight_scale)
            else:
                raise ValueError(f"Invalid quant type: {cfg.quant_type}")
            logger.info(f"Dequantized weight: {dequant_weight.shape}, {dequant_weight.dtype}")

            del model[tgt]
            del model[tgt.replace(".weight_packed", ".weight_scale")]
            if cfg.quant_type == "nvfp4":
                del model[tgt.replace(".weight_packed", ".weight_global_scale")]
            model[tgt.replace(".weight_packed", ".weight")] = dequant_weight

        if output_dir is not None:
            output_path = output_dir / file.name
            logger.info(f"Saving to {output_path}")
            save_file(model, output_path)

    # move other files from input dir to output dir
    for file in input_dir.glob("*"):
        if file.is_file() and all(not file.name.endswith(x) for x in [".safetensors", ".pt"]):
            shutil.copy(file, output_dir / file.name)

    # read config file and remove quantization related configs
    logger.info(f"Removing quantization related configs")
    model_cfg_path = output_dir / "config.json"
    with open(model_cfg_path, "r") as f:
        model_cfg = json.load(f)
    del model_cfg["quantization_config"]
    with open(model_cfg_path, "w") as f:
        json.dump(model_cfg, f, indent=4)
    logger.info(model_cfg)

    # Verify
    if cfg.verify_with_transformers:
        logger.info(f"Verifying with transformers")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        logger.info(f"Model: {model}")
    logger.info(f"Done")


if __name__ == "__main__":
    main()
