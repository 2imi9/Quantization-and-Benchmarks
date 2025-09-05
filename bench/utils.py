import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def write_jsonl(samples: list[dict], path: str | Path) -> None:
    with open(path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False, cls=DateTimeEncoder) + '\n')

def write_json(data: dict, path: str | Path) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

def save_config(cfg: DictConfig, path: str | Path) -> None:
    yaml_str = OmegaConf.to_yaml(cfg)
    with open(path, "w") as f:
        f.write(yaml_str)

def read_jsonl(path: str | Path) -> list[dict]:
    samples = []
    with open(path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples
