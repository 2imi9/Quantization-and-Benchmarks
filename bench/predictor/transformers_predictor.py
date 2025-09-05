import math

import torch
from tqdm import tqdm
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BasePredictor, TestSample, get_compute_capability


class TransformersPredictor(BasePredictor):

    def __init__(self, model_name: str, tokenizer: AutoTokenizer, cfg: DictConfig) -> None:
        super().__init__(model_name, tokenizer, cfg)
        major, minor = get_compute_capability()
        if major <= 7.0 and minor <= 0.0:
            logger.warning("GPU compute capability is less than 7.0, disabling mem efficient SDP")
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self._bsz = cfg.batch_size

    def _get_output_token_ids(self, input_ids: torch.Tensor, generated_ids: torch.Tensor, num_return_sequences: int = 1) -> list:
        output_token_ids = []
        bsz, context_len = input_ids.size()
        for i in range(bsz):
            output_token_ids.append(generated_ids[i, context_len:].tolist())
        return output_token_ids

    def predict(self, samples: list[TestSample], sample_params: DictConfig) -> list[TestSample]:
        num_samples = len(samples)
        num_batch = int(math.ceil(num_samples / self._bsz))
        with torch.no_grad():
            for bid in tqdm(range(num_batch)):
                batch_samples = samples[bid * self._bsz:(bid + 1) * self._bsz]
                batch_prompts = [s.prompt for s in batch_samples]
                inputs = self._tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self._model.device)
                generated_ids = self._model.generate(
                    **inputs,
                    temperature=sample_params.temperature,
                    top_p=sample_params.top_p,
                    top_k=sample_params.top_k,
                    max_length=sample_params.max_sequence_length,
                    do_sample=sample_params.do_sample,
                    num_return_sequences=sample_params.num_return_sequences
                )
                output_ids = self._get_output_token_ids(inputs.input_ids, generated_ids, sample_params.num_return_sequences)
                preds = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                assert len(preds) == len(batch_samples), f"Batch size mismatch: {len(preds)} vs {len(batch_samples)}"
                for sample in batch_samples:
                    sample.preds.clear()  # Clear previous outputs
                for sample, pred in zip(batch_samples, preds):
                    logger.debug(f"{pred}")
                    sample.preds.append(pred)
        return samples