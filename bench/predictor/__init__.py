from omegaconf import DictConfig
from transformers import AutoTokenizer

from .base import TestSample, BasePredictor
from .vllm_predictor import VLLMPredictor
from .transformers_predictor import TransformersPredictor


def get_predictor(cfg: DictConfig, tokenizer: AutoTokenizer) -> BasePredictor:
    """
    Factory function to get the appropriate predictor
    """
    if cfg.eval_predictor == "transformers":
        return TransformersPredictor(
            model_name=cfg.model_name,
            tokenizer=tokenizer,
            cfg=cfg.predictor_conf[cfg.eval_predictor]
        )
    elif cfg.eval_predictor == "vllm":
        return VLLMPredictor(
            model_name=cfg.model_name,
            tokenizer=tokenizer,
            cfg=cfg.predictor_conf[cfg.eval_predictor]
        )
    else:
        raise ValueError(f"Unknown predictor: {cfg.predictor}")
