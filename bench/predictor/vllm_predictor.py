import math

from loguru import logger
from omegaconf import DictConfig
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .base import BasePredictor, TestSample, get_compute_capability


class VLLMPredictor(BasePredictor):

    def __init__(self, model_name: str, tokenizer: AutoTokenizer, cfg: DictConfig) -> None:
        super().__init__(model_name, tokenizer, cfg)
        self._eos_token_id = tokenizer.eos_token_id
        self._pad_token_id = tokenizer.pad_token_id

        # Extract pipeline parallelism config from cfg
        pipeline_parallel_size = getattr(cfg, 'pipeline_parallel_size', 1)
        tensor_parallel_size = getattr(cfg, 'tensor_parallel_size', 1)
        gpu_memory_utilization = getattr(cfg, 'gpu_memory_utilization', 0.9)
        max_seq_len = getattr(cfg, 'max_seq_len', 4096)
        max_num_batched_tokens = getattr(cfg, 'max_num_batched_tokens', 16384)
        enable_prefix_caching = getattr(cfg, 'enable_prefix_caching', False)
        quantization = getattr(cfg, 'quantization', None)
        enforce_eager = getattr(cfg, 'enforce_eager', False)
        cpu_offload_gb = getattr(cfg, 'cpu_offload_gb', 0)
        if quantization is not None:
            logger.info(f"Using quantization: {quantization}")
        if cpu_offload_gb > 0:
            logger.info(f"Using CPU offload with {cpu_offload_gb}GB")

        major, minor = get_compute_capability()
        if major <= 7.0 and minor <= 0.0:
            logger.warning("GPU compute capability is less than 7.0, Pipeline parallelism is not supported and BF16 is not supported")
            assert pipeline_parallel_size == 1, "Pipeline parallelism is not supported for GPUs with compute capability less than 7.0"
            dtype = "half"
        else:
            dtype = "auto"

        self._model = LLM(
            model=model_name,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_seq_len,
            # Disable prefix cache for better memory efficiency
            enable_prefix_caching=enable_prefix_caching,
            # Enable dynamic batching for efficient inference
            max_num_batched_tokens=max_num_batched_tokens,
            # Maximum number of sequences to generate in parallel
            max_num_seqs=cfg.get('max_num_seqs', 8),
            quantization=quantization,
            # Additional optimizations
            trust_remote_code=True,
            dtype=dtype,
            task="generate",
            enforce_eager=enforce_eager,
            cpu_offload_gb=cpu_offload_gb
        )

    def predict(self, samples: list[TestSample], sample_params: DictConfig) -> list[TestSample]:
        num_truncated = 0
        num_samples = len(samples)

        # Prepare all prompts for dynamic batching
        all_prompt_token_ids = list()
        tokenizer = self._model.get_tokenizer()
        for sample in samples:
            p = sample.prompt
            token_ids = tokenizer.encode(p)
            all_prompt_token_ids.append(token_ids[-sample_params.max_sequence_length:])

        # Generate using vLLM with dynamic batching
        sampling_params = SamplingParams(
            n=sample_params.num_return_sequences,
            temperature=sample_params.temperature,
            top_p=sample_params.top_p,
            top_k=sample_params.top_k,
            max_tokens=sample_params.max_sequence_length, # Maximum number of tokens to generate per output sequence.
            presence_penalty=sample_params.presence_penalty,
            ignore_eos=False
        )

        logger.info(f"Starting generation for {num_samples} samples with dynamic batching")
        # vLLM handles dynamic batching automatically
        outputs = self._model.generate(prompt_token_ids=all_prompt_token_ids, sampling_params=sampling_params)

        # Extract predictions
        for sample, output in zip(samples, outputs):
            # vLLM outputs contain the generated text
            for i, o in enumerate(output.outputs):
                generated_text = o.text
                sample.preds.append(generated_text)
                # Check if the generated token ids is truncated
                if o.token_ids[-1] in {self._pad_token_id, self._eos_token_id}:
                    num_truncated += 1

        logger.info(f"Generated predictions for {num_samples} samples")
        logger.info(f"{num_truncated}/{num_samples} samples are truncated!!")
        return samples
