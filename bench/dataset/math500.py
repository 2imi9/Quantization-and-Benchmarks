import re
from typing import List, Dict, Any

from loguru import logger
from datasets import load_dataset, Dataset

from .base import BaseDataset


class Math500Dataset(BaseDataset):

    DATASET_NAME = "HuggingFaceH4/MATH-500"

    @classmethod
    def load(cls, debug: bool = False) -> Dataset:
        """Load Math-500 dataset"""
        dataset = load_dataset(cls.DATASET_NAME, split="test", trust_remote_code=True)
        if debug:
            # Use only first 5 samples for debugging
            dataset = dataset.select(range(5))
            print(f"Debug mode: Using {len(dataset)} samples from Math-500")
        else:
            print(f"Loaded Math-500 dataset with {len(dataset)} samples")
        return dataset

    @classmethod
    def measure(cls, predictions: List) -> Dict[str, Any]:
        """Measure Math-500 performance using Math-Verify
        TODO: The current measurement doesn't support well the following cases, leading to under-estimated performance
        1. Matrix representation
        2. Multiple answers seperated by commas, e.g., \boxed{0}, \boxed{1}
        """
        from math_verify import parse, verify
        from math_verify import LatexExtractionConfig

        num_correct, num_parse_failed, num_eval_faild = 0, 0, 0
        total = len(predictions)

        config = LatexExtractionConfig()
        for pred_data in predictions:
            pred_answer = pred_data.preds[0].strip()
            gold_answer = pred_data.item['solution'].strip()

            # Localized the last answer encapsulated by \boxed using regex
            parsed_pred_answer = cls._parse_pred_answer(pred_answer)
            if len(parsed_pred_answer) == 0:
                num_parse_failed += 1
                # logger.warning(f"Skipping prediction with no boxed answer: {pred_answer}")
                continue

            try:
                pred_parsed = parse(parsed_pred_answer)
                if len(pred_parsed) == 0:
                    pred_parsed.append(parsed_pred_answer)  # Ensure we have a list
                gold_parsed = parse(gold_answer, extraction_config=[config])
                if len(gold_parsed) == 0:
                    parsed_gold_answer = cls._parse_pred_answer(gold_answer)
                    gold_parsed = parse(parsed_gold_answer)
                if verify(gold_parsed, pred_parsed):
                    num_correct += 1
                else:
                    logger.warning(f"Incorrect prediction: {parsed_pred_answer} (expected: {gold_parsed}) for Question: {pred_data.item['problem']}")
            except Exception as e:
                num_eval_faild += 1
                logger.error(f"Failed to parse prediction '{pred_answer}' or gold answer '{gold_answer}': {e}")
                continue

        accuracy = (num_correct / total) * 100 if total > 0 else 0
        baseline = 97.0
        diff_from_baseline = accuracy - baseline
        result_dict = {
            'accuracy': round(accuracy, 2),
            'correct_answers': num_correct,
            'parse_failures': num_parse_failed,
            'eval_failures': num_eval_faild,
            'total_samples': total,
            'baseline': baseline,
            'diff_from_baseline': round(diff_from_baseline, 2),
            'performance_vs_baseline': 'above' if diff_from_baseline > 0 else 'below' if diff_from_baseline < 0 else 'equal'
        }
        logger.info(result_dict)
        return result_dict

    @classmethod
    def _parse_pred_answer(cls, pred_answer: str) -> str:
        """Parse the predicted answer to extract the last boxed answer
        Args:
            pred_answer (str): The predicted answer string, e.g., "xxxxxxx \\boxed{answer} yyyyyy"
        Returns:
            str: The extracted boxed answer or an empty string if not found, e.g., "\\boxed{answer}"
        """

        # find the last \boxed in the prediction
        beg_idx = pred_answer.rfind('\\boxed{')
        end_idx = pred_answer.rfind('}')
        if beg_idx == -1 or end_idx == -1 or beg_idx >= end_idx:
            return ""

        res = pred_answer[beg_idx+len("\\boxed"):end_idx+1].strip()
        # Check brackets balance of "{}". Truncate res to the point where "{}" are balanced
        res = truncate_first_balanced(res)
        return "\\boxed" + res


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
