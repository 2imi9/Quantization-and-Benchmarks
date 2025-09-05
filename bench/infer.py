import importlib
from pathlib import Path
from typing import Callable

import hydra
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv
from transformers import AutoTokenizer
from datasets import Dataset as HF_Dataset
from omegaconf import DictConfig

from dataset import DATASETS
from utils import write_jsonl, write_json, save_config
from compute_metric import post_process_predictions
from predictor import TestSample, get_predictor

load_dotenv()


def sanity_check(cfg: DictConfig) -> None:
    ds = cfg.eval_dataset
    assert ds in DATASETS, f"{ds} is not available"
    if cfg.eval_dataset_config is None:
        cfg.eval_dataset_config = ds
    assert cfg.eval_dataset_config in cfg.dataset, f"Missing model params for {cfg.eval_dataset_config}"

    assert cfg.output_dir is not None, "Output directory must be specified"
    output_dir = Path(cfg.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)


def load_chat_template(template_name: str) -> Callable:
    """
    Load the chat template function by name.
    """
    series, fn_name = template_name.split(".")
    chat_templates = importlib.import_module(f"chat_template.{series}")
    return getattr(chat_templates, fn_name)


def prepare_test_samples(
    dataset: HF_Dataset,
    tokenizer: AutoTokenizer,
    chat_template_fn: Callable,
    cfg: DictConfig
) -> list[TestSample]:
    chat_samples = list()
    for i, s in tqdm(enumerate(dataset)):
        msgs = chat_template_fn(s)
        # TODO: Refactor
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.get("enable_thinking", False), # For Qwen, switches between thinking and non-thinking modes. Default is True.
            reasoning_effort=cfg.get("reasoning_effort", "medium"), # For GPT-OSS, specifies the reasoning effort. Default is "high".
        )
        chat_samples.append(TestSample(prompt=prompt, item=s))
        if i % 100:
            logger.info(prompt)
    return chat_samples


@hydra.main(config_path="conf", config_name="qwen3_4B_bf16", version_base=None)
def main(cfg: DictConfig) -> None:
    sanity_check(cfg)

    ds = DATASETS[cfg.eval_dataset].load(debug=cfg.debug)
    logger.info(f"Loading {cfg.eval_dataset} with {len(ds)} samples")

    ds_cfg = cfg.dataset[cfg.eval_dataset_config]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, padding_side="left")

    chat_template_fn = load_chat_template(ds_cfg.template_fn)
    test_samples = prepare_test_samples(ds, tokenizer, chat_template_fn, ds_cfg)

    predictor = get_predictor(cfg, tokenizer)
    pred_samples = predictor.predict(test_samples, ds_cfg)

    # Save predictions and config
    op = Path(cfg.output_dir)
    write_jsonl([s.to_dict() for s in pred_samples], op / "predictions.jsonl")
    save_config(cfg, op / "config.yaml")

    # Compute and save metrics
    pred_samples = post_process_predictions(ds_cfg, pred_samples)
    metrics = DATASETS[cfg.eval_dataset].measure(pred_samples)
    write_json(metrics, op / "metrics.json")

    logger.info(f"Evaluation metrics: {metrics}")


if __name__ == '__main__':
    main()
