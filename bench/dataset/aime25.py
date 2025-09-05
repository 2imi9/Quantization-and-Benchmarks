from typing import List, Dict, Any

import numpy as np
from loguru import logger
from datasets import load_dataset, Dataset, \
    concatenate_datasets

from .base import BaseDataset


class AIME25Dataset(BaseDataset):

    DATASET_NAME = "opencompass/AIME2025"

    @classmethod
    def load(cls, debug: bool = False) -> Dataset:
        """Load AIME 2025 dataset - both parts I and II"""
        dataset1 = load_dataset(cls.DATASET_NAME, "AIME2025-I", split="test")
        dataset2 = load_dataset(cls.DATASET_NAME, "AIME2025-II", split="test")
        # Combine both datasets
        dataset = concatenate_datasets([dataset1, dataset2])
        if debug:
            dataset = dataset.select(range(3))
            print(f"Debug mode: Using {len(dataset)} samples from AIME 2025")
        else:
            print(f"Loaded AIME 2025 dataset with {len(dataset)} samples (Part I: {len(dataset1)}, Part II: {len(dataset2)})")
        return dataset

    @classmethod
    def measure(cls, predictions: List) -> Dict[str, Any]:
        """Measure AIME 2025 performance using Math-Verify"""
        from math_verify import parse, verify
        from math_verify import LatexExtractionConfig

        num_parse_failed, num_eval_failed = 0, 0
        total_questions = len(predictions)
        total_attempts = 0
        total_correct_attempts = 0
        question_results = []

        config = LatexExtractionConfig()

        for pred_data in predictions:
            gold_answer = str(pred_data.item['answer'])

            # Analyze all attempts for this question
            question_correct_count = 0
            question_attempts = len(pred_data.preds)
            total_attempts += question_attempts

            for attempt_idx, pred_answer in enumerate(pred_data.preds):
                pred_answer = pred_answer.strip()
                # Extract the last boxed answer using same logic as Math500
                parsed_pred_answer = cls._parse_pred_answer(pred_answer)
                if len(parsed_pred_answer) == 0:
                    num_parse_failed += 1
                    continue
                try:
                    # Use math-verify like Math500
                    pred_parsed = parse(parsed_pred_answer)
                    if len(pred_parsed) == 0:
                        pred_parsed.append(parsed_pred_answer)  # Ensure we have a list

                    # For AIME, gold answer is usually just an integer, wrap it in \boxed{}
                    if gold_answer.isdigit():
                        gold_answer_boxed = f"\\boxed{{{gold_answer}}}"
                    else:
                        gold_answer_boxed = gold_answer

                    gold_parsed = parse(gold_answer_boxed, extraction_config=[config])
                    if len(gold_parsed) == 0:
                        # Try parsing the raw answer
                        gold_parsed = parse(gold_answer)
                        if len(gold_parsed) == 0:
                            # If still fails, add as string
                            gold_parsed.append(gold_answer)
                    if verify(gold_parsed, pred_parsed):
                        question_correct_count += 1
                        total_correct_attempts += 1
                except Exception as e:
                    num_eval_failed += 1
                    logger.error(f"Failed to parse prediction '{pred_answer}' or gold answer '{gold_answer}': {e}")
                    continue

            # Record question results
            question_results.append({
                'correct_attempts': question_correct_count,
                'total_attempts': question_attempts,
                'pass_at_1': question_correct_count > 0,  # At least one correct
                'pass_at_1_n': pass_at_k(c=question_correct_count, n=question_attempts, k=1) # https://arxiv.org/pdf/2107.03374
            })

        # Calculate pass@1 and pass@1_n
        pass_at_1 = np.mean([q['pass_at_1'] for q in question_results])
        pass_at_1_n = np.mean([q['pass_at_1_n'] for q in question_results])
        return {
            'pass_at_1': pass_at_1,
            'pass_at_1_n': pass_at_1_n,
            'total_questions': total_questions,
            'total_attempts': total_attempts,
            'num_parse_failed': num_parse_failed,
            'num_eval_failed': num_eval_failed,
        }

    @classmethod
    def _parse_pred_answer(cls, pred_answer: str) -> str:
        """Parse the predicted answer to extract the last boxed answer
        Args:
            pred_answer (str): The predicted answer string, e.g., "xxxxxxx \\boxed{answer} yyyyyy"
        Returns:
            str: The extracted boxed answer or an empty string if not found, e.g., "\\boxed{answer}"
        """
        # Find the last \boxed in the prediction
        beg_idx = pred_answer.rfind('\\boxed{')
        end_idx = pred_answer.rfind('}')
        if beg_idx == -1 or end_idx == -1 or beg_idx >= end_idx:
            return ""

        res = pred_answer[beg_idx+len("\\boxed"):end_idx+1].strip()
        # Check brackets balance of "{}". Truncate res to the point where "{}" are balanced
        res = truncate_first_balanced(res)
        return "\\boxed" + res

    @classmethod
    def _extract_numeric(cls, text: str) -> int:
        """Extract integer from text"""
        # Remove boxed wrapper if present
        if text.startswith('\\boxed{') and text.endswith('}'):
            text = text[7:-1]
        
        # Handle common answer formats
        text = text.strip()
        
        # Try direct integer conversion first
        if text.isdigit():
            num = int(text)
            return num if 0 <= num <= 999 else None
        
        # Look for integers in the text
        import re
        # Find all numbers, prioritize 3-digit, then 2-digit, then 1-digit
        numbers = re.findall(r'\d+', text)
        if numbers:
            # Try to find a number in valid AIME range
            for num_str in numbers:
                num = int(num_str)
                if 0 <= num <= 999:
                    return num
        
        return None


def truncate_first_balanced(s: str) -> str:
    """
    Given s starting with '{', return s[:i+1] where
    i is the first index such that all opened '{' are closed.
    If braces never balance, return the entire s.
    """
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

        # once we're back to zero nesting, truncate here
        if depth == 0:
            return s[: i + 1]

    # never balanced → return whole string
    return s


def pass_at_k(c, n: int, k: int) -> float:
    """Algo from https://arxiv.org/pdf/2107.03374"""
    if n - c < k:
        return 1.0

    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
