import re

import torch
import torch.nn as nn

# avoid division by zero when calculating scale
# TODO: align this value with NVIDIA's assumptions (current value is a guess)
EPS = 1e-12


class FP8Linear(nn.Linear):

    BLOCK_SIZE = 128

    def _quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        assert shape[-1] % self.BLOCK_SIZE == 0, f"x.shape[-1] {shape[-1]} must be divisible by BLOCK_SIZE {self.BLOCK_SIZE}"
        _x = x.reshape(shape[0], shape[1] // self.BLOCK_SIZE, self.BLOCK_SIZE)
        # AMAX
        amax = _x.abs().max(dim=-1, keepdim=True)
        amax = amax.to(torch.float64)
        scale = torch.finfo(torch.float8_e4m3fn).max / torch.clamp(amax, min=EPS)
        scale = scale.to(torch.float32)
        # Quantize
        _x = _x / scale
        # FP32 to e4m3fn
        _x = _x.to(torch.float8_e4m3fn)
        # Reshape
        _x = _x.reshape(shape)
        return _x, scale

    def _dequantize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        _x = x.reshape(shape[0], shape[1] // self.BLOCK_SIZE, self.BLOCK_SIZE)
        # fp8 e4m3 to fp32
        _x = _x.to(torch.float32)
        _x = _x * scale
        _x = _x.reshape(shape).to(torch.bfloat16)
        return _x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert x to fp8 and back to bf16
        # block size 128
        # element dtype torch.float8_e4m3fn
        # scale dtype torch.float32
        shape = x.shape
        _x = x.reshape(-1, shape[-1])
        quantized_x, scale = self._quantize(x)
        dequantized_x = self._dequantize(quantized_x, scale)

        w = self.weight
        output = torch.mm(dequantized_x, w.t())
        output = output.reshape(*shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias
        return output


class MXFP8Linear(nn.Linear):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MXFP4Linear(nn.Linear):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class NVFP4Linear(nn.Linear):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


def get_layer_parent_by_name(model: nn.Module, input_name: str) -> nn.Module:
    parent_name = input_name.rsplit(".", 1)[:-1]
    if len(parent_name) == 0:  # parent is model itself
        return model
    else:
        parent_name = parent_name[0]

    for name, module in model.named_modules():
        if parent_name == name:
            return module
    raise ValueError(f"Layer {input_name} not found in model")


def replace_linear_with_quantized_linear(model: nn.Module, activation_quantization_type: str, target_patterns: list[str]) -> nn.Module:
    patterns = [re.compile(pattern) for pattern in target_patterns]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(pattern.match(name) for pattern in patterns):
                if activation_quantization_type == "fp8":
                    linear = FP8Linear(module.in_features, module.out_features, module.bias is not None)
                elif activation_quantization_type == "mxfp8":
                    linear = MXFP8Linear(module.in_features, module.out_features, module.bias is not None)
                elif activation_quantization_type == "mxfp4":
                    linear = MXFP4Linear(module.in_features, module.out_features, module.bias is not None)
                elif activation_quantization_type == "nvfp4":
                    linear = NVFP4Linear(module.in_features, module.out_features, module.bias is not None)
                else:
                    raise ValueError(f"Invalid activation quantization type: {activation_quantization_type}")
                linear.weight = module.weight
                linear.bias = module.bias
                # replace the module with the quantized linear module
                attribute_name = name.rsplit(".", 1)[-1]
                parent_of_module = get_layer_parent_by_name(model, name)
                setattr(parent_of_module, attribute_name, linear)
    return model
