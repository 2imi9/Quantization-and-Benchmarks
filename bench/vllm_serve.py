import os

import hydra
from loguru import logger
from dotenv import load_dotenv
from omegaconf import DictConfig

load_dotenv()

@hydra.main(config_path="conf", config_name="qwen3_30B_A3B_Instruct_bf16", version_base=None)
def main(cfg: DictConfig) -> None:
    
    API_TOKEN = os.getenv("API_SERVER_TOKEN")
    server_cfg = cfg.predictor_conf.vllm
    logger.info(f"Server config: {server_cfg}")

    command = f"""vllm serve {cfg.model_name} --port 8000 --host 0.0.0.0 \
--pipeline-parallel-size {server_cfg.pipeline_parallel_size} \
--tensor-parallel-size {server_cfg.tensor_parallel_size} \
--gpu-memory-utilization {server_cfg.gpu_memory_utilization} \
--cpu-offload-gb {server_cfg.cpu_offload_gb} \
--max-seq-len {server_cfg.max_seq_len} \
--max-num-batched-tokens {server_cfg.max_num_batched_tokens} \
--max-num-seqs {server_cfg.max_num_seqs} \
--max-model-len {server_cfg.max_seq_len} \
--enforce_eager \
--no-enable-prefix-caching \
--enable-auto-tool-choice \
--tool-call-parser hermes \
--api-key {API_TOKEN}"""

    logger.info(f"Running command: {command}")
    os.system(command)


if __name__ == "__main__":
    main()
