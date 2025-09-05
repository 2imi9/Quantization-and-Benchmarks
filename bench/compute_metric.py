import argparse
import importlib
from pathlib import Path

from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer
from omegaconf import OmegaConf, DictConfig

from dataset import DATASETS
from predictor import TestSample
from utils import read_jsonl, write_json


def post_process_predictions(ds_cfg: DictConfig, samples: list[TestSample]) -> list[TestSample]:
    """
    Post-process predictions to trim reasoning tokens if enabled.
    """
    if ds_cfg.get("enable_thinking", False) and ds_cfg.get("trim_reasoning_tokens_fn", None) is not None:
        logger.info(f"Trim reasoning tokens with {ds_cfg.trim_reasoning_tokens_fn}")
        series, fn_name = ds_cfg.trim_reasoning_tokens_fn.split(".")
        module = importlib.import_module(f"chat_template.{series}")
        fn = getattr(module, fn_name)
        for idx, s in enumerate(samples):
            if idx == 0:
                logger.debug(f"With reasoning tokens:\n{s.preds[0]}")
            s.preds = [fn(p) for p in s.preds]
            if idx == 0:
                logger.debug(f"Without reasoning tokens:\n{s.preds[0]}")
    return samples


def main(cfg: DictConfig, path: Path, preds: list[TestSample], check_truncation: bool = False) -> None:
    dataset_cls = DATASETS[cfg.eval_dataset]
    ds_cfg = cfg.dataset[cfg.eval_dataset_config]
    preds = post_process_predictions(ds_cfg, preds)

    metrics = dataset_cls.measure(preds)
    write_json(metrics, path / "metrics.json")

    if check_truncation:
        logger.info("Checking truncation...")
        seq_lens = list()
        num_truncated = 0
        max_seq_len = ds_cfg.max_sequence_length
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        for sample in tqdm(preds, desc="Checking truncation"):
            prompt = sample.prompt + sample.preds[0]
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            seq_lens.append(len(tokens))
            if len(tokens) >= max_seq_len:
                num_truncated += 1
        logger.info(f"Max sequence length: {max(seq_lens)}")
        logger.info(f"Min sequence length: {min(seq_lens)}")
        logger.info(f"Mean sequence length: {sum(seq_lens) / len(seq_lens)}")
        logger.info(f"Median sequence length: {sorted(seq_lens)[len(seq_lens) // 2]}")
        logger.info(f"90th percentile sequence length: {sorted(seq_lens)[int(0.9 * len(seq_lens))]}")
        logger.info(f"95th percentile sequence length: {sorted(seq_lens)[int(0.95 * len(seq_lens))]}")
        logger.info(f"99th percentile sequence length: {sorted(seq_lens)[int(0.99 * len(seq_lens))]}")
        logger.info(f"Number of truncated samples: {num_truncated}/{len(preds)}, Ratio: {num_truncated / len(preds)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute metrics")
    parser.add_argument("--conf", "-c", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--check-truncation", "-t", action="store_true", help="Check truncation")
    args = parser.parse_args()
    path = Path(args.conf)
    conf = path / "config.yaml"
    pred_path = path / "predictions.jsonl"
    if not conf.exists() or not pred_path.exists():
        raise FileNotFoundError(f"Configuration file or Prediction file not found at {conf}")
    cfg = OmegaConf.load(conf)
    preds = [TestSample.from_dict(s) for s in read_jsonl(pred_path)]
    main(cfg, path, preds, args.check_truncation)  # Assuming main is defined in the imported module
