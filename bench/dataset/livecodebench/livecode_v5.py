import re
import zlib
import json
import base64
import pickle
from typing import List, Dict, Any

from loguru import logger
from datasets import load_dataset, Dataset

from ..base import BaseDataset
from .codegen_metrics import extract_code, codegen_metrics, extract_code_v2


class LiveCodeBenchV5Dataset(BaseDataset):
    
    DATASET_NAME = "livecodebench/code_generation_lite"
    
    @classmethod
    def load(cls, debug: bool = False) -> Dataset:
        """Load LiveCodeBench V5 dataset"""
        # Try multiple approaches to load the real dataset
        dataset = None
        
        # Try 1: Default loading
        try:
            dataset = load_dataset(cls.DATASET_NAME, split="test")
            print(f"Loaded {len(dataset)} samples from default")
        except Exception as e1:
            print(f"Default loading failed: {e1}")
            
            # Try 2: Different dataset name
            try:
                dataset = load_dataset("livecodebench/code_generation", split="test")
                print(f"Loaded {len(dataset)} samples from code_generation")
            except Exception as e2:
                print(f"Alternative dataset failed: {e2}")
                
                # Try 3: With streaming
                try:
                    dataset = load_dataset(cls.DATASET_NAME, split="test", streaming=False)
                    print(f"Loaded {len(dataset)} samples with streaming=False")
                except Exception as e3:
                    print(f"All attempts failed: {e1}, {e2}, {e3}")
                    print("Using fallback dataset")
                    return cls._create_fallback_dataset()
        
        # Filter for Qwen3 subset (2410-2502)
        filtered_dataset = []
        for sample in dataset:
            contest_date = sample.get("contest_date", "")
            if contest_date and cls._is_qwen3_range(contest_date):
                filtered_dataset.append(sample)
        
        if filtered_dataset:
            dataset = Dataset.from_list(filtered_dataset)
            print(f"Qwen3 subset (2410-2502): {len(dataset)} samples")
        else:
            print("No samples found in Qwen3 date range, using full dataset")
        
        if debug:
            dataset = dataset.select(range(min(20, len(dataset))))
            print(f"Debug mode: {len(dataset)} samples")
        
        return dataset

    @classmethod
    def _is_qwen3_range(cls, date_str: str) -> bool:
        """Check if date is in Qwen3 range (2024-10-01 to 2025-02-01)"""
        try:
            if len(date_str) == 8:  # YYYYMMDD
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            return "2024-10-01" <= date_str <= "2025-02-01"
        except:
            return True

    @classmethod
    def measure(cls, predictions: List) -> Dict[str, Any]:
        """Official LiveCodeBench evaluation - Pass@1 only"""

        # Prepare evaluation data
        all_code_snippets = []
        all_evaluation_samples = []

        for pidx, pred_data in enumerate(predictions):
            response = pred_data.preds[0].strip()
            item = pred_data.item

            code = extract_code(response)
            if pidx == 0:
                logger.debug(f"Extracted Code:\n{code}")
            all_code_snippets.append([code])
            test_cases = json.loads(item['public_test_cases'])
            private_test_cases = cls._parse_private_test_cases(item["private_test_cases"])
            metadata = json.loads(item['metadata'])
            evaluation_sample = {
                "input_output": json.dumps({
                    "inputs": [tc['input'] for tc in test_cases + private_test_cases],
                    "outputs": [tc['output'] for tc in test_cases + private_test_cases],
                    "fn_name": metadata.get('function_name', None),
                })
            }
            all_evaluation_samples.append(evaluation_sample)

        if not all_code_snippets:
            logger.error("No samples to measure")
            return {'pass@1': 0.0, 'total_problems': len(predictions)}

        try:
            # Official evaluation - Pass@1 only
            metrics, _ = codegen_metrics(
                all_evaluation_samples,
                all_code_snippets,
                k_list=[1],
                num_process_evaluate=16,
                timeout=10
            )
            pass_at_1 = metrics.get('pass@1', 0.0) * 100
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {'pass@1': 0.0, 'total_problems': len(predictions)}

        metrics = {
            'pass@1': round(pass_at_1, 2),
            'total_problems': len(predictions),
            'baseline': 64.8,  # Qwen3 baseline
            'diff_from_baseline': round(pass_at_1 - 64.8, 2)
        }
        logger.info(metrics)
        metrics['code_snippets'] = all_code_snippets
        return metrics

    @classmethod
    def _extract_function_name(cls, code: str) -> str:
        """Extract function name from code"""
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else 'solution'

    @classmethod
    def _create_fallback_dataset(cls) -> Dataset:
        """Create fallback dataset"""
        data = [{
            "question_title": "Two Sum",
            "question_content": "Find two numbers that add up to target.",
            "contest_date": "2024-10-15"
        }]
        return Dataset.from_list(data)

    @classmethod
    def _parse_private_test_cases(cls, private_test_cases: str) -> List[Dict[str, Any]]:
        """Parse private test cases from string to list of dicts"""
        try:
            results = json.loads(private_test_cases)
        except Exception as e:  # noqa: F841
            results = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(private_test_cases.encode(
                            'utf-8'))  # type: ignore
                    )))
        return results
