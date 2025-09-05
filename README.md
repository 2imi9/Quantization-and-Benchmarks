# Benchmark Low-Bit LLMs

## Setup

Build and launch docker containers
```
cd ./dockers
sudo ./build_docker.sh ./rtx_3080_Dockerfil
sudo ./launch_docker.sh jiaqi_dev
```

## Testing LLMs

1. Create a configuration for dataset and model, e.g., `/bench/conf/qwen3_4B_bf16.yaml`
2. Generate samples
```
cd bench
python3 ./infer.py eval_dataset=mmlu-redux output_dir=/app/outputs/qwen3_4B_bf16_mmlu_redux
```
3. Compute Metrics
```
python3 ./compute_metric.py -c /app/outputs/qwen3_4B_bf16_mmlu_redux
```

## Onboard Dataset

1. Implement dataset in `./bench/dataset` based on this abstract class
```
import abc
from datasets import Dataset


class BaseDataset(abc.ABC):

    DATASET_NAME = ""

    @classmethod
    @abc.abstractmethod
    def load(cls, debug: bool = False) -> Dataset:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def measure(cls, predictions: list) -> dict:
        pass
```

2. Register the dataset in `./bench/dataset/__init__.py`

## Leaderboard

### Qwen3-4B-Bf16 (Non-thinking)

| Dataset    | Number | Reference | Conf                            |
|----------- |--------| --------- | ---------------------------     |
| MMLU-Redux | 72.2   | 77.3      |  qwen3_4B_bf16.yaml             |
