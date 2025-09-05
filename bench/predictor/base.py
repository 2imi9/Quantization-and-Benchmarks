import abc
import subprocess
from dataclasses import dataclass, field

from omegaconf import DictConfig
from transformers import AutoTokenizer


@dataclass
class TestSample:
    prompt: str
    item: dict
    preds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "item": self.item,
            "preds": self.preds
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TestSample':
        return cls(
            prompt=data.get("prompt", ""),
            item=data.get("item", {}),
            preds=data.get("preds", [])
        )


class BasePredictor(abc.ABC):

    def __init__(self, model_name: str, tokenizer: AutoTokenizer, cfg: DictConfig) -> None:
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._cfg = cfg

    @abc.abstractmethod
    def predict(self, samples: list[TestSample], sample_params: DictConfig) -> list[TestSample]:
        raise NotImplementedError


def get_gpu_info() -> dict[str, str]:
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = {}
            for i, line in enumerate(lines):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        compute_cap = parts[1].strip()
                        gpu_info[f"GPU_{i}"] = {
                            "name": name,
                            "compute_cap": compute_cap
                        }
            return gpu_info
    except Exception as e:
        print(f"Warning: Could not get GPU info: {e}")
    return {}


def get_compute_capability() -> tuple[int, int]:
    """Get the minimum compute capability across all GPUs"""
    gpu_info = get_gpu_info()
    if not gpu_info:
        return (7, 0)  # Default to v100 compatibility

    min_major = 8
    min_minor = 0

    for gpu_id, info in gpu_info.items():
        try:
            major, minor = map(int, info['compute_cap'].split('.'))
            if major < min_major or (major == min_major and minor < min_minor):
                min_major = major
                min_minor = minor
        except ValueError:
            continue

    return (min_major, min_minor)
