import re

from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, concatenate_datasets, \
    Dataset

from .base import BaseDataset


class MmluReduxDataset(BaseDataset):

    DATASET_NAME = "edinburgh-dawg/mmlu-redux"

    CONFIGS = [
        'anatomy', 'business_ethics', 'clinical_knowledge', 'college_chemistry', 'college_computer_science',
        'college_mathematics', 'college_medicine', 'college_physics', 'econometrics', 'electrical_engineering',
        'formal_logic', 'global_facts', 'high_school_chemistry', 'high_school_mathematics', 'high_school_physics',
        'high_school_statistics', 'human_aging', 'logical_fallacies', 'machine_learning', 'miscellaneous', 'philosophy',
        'professional_accounting', 'public_relations', 'virology', 'conceptual_physics', 'high_school_us_history',
        'astronomy', 'high_school_geography', 'high_school_macroeconomics', 'professional_law'
    ]

    @classmethod
    def load(cls, debug: bool = False) -> Dataset:
        configs = cls.CONFIGS[:1] if debug else cls.CONFIGS
        _datasets = [load_dataset(cls.DATASET_NAME, conf, split="test") for conf in tqdm(configs)]

        def __add_fields(sample, name):
            sample['config_name'] = name
            return sample

        for i, (ds, conf) in enumerate(zip(_datasets, configs)):
            _datasets[i] = ds.map(lambda x: __add_fields(x, conf))
        dataset = concatenate_datasets(_datasets)
        return dataset

    @classmethod
    def measure(cls, predictions: list) -> dict:
        fail_cnt = 0
        num_samples = len(predictions)
        correct_cnt, incorrect_cnt = 0, 0
        for sample in predictions:
            if sample.item['error_type'] == 'no_correct_answer':
                num_samples -= 1
                continue
            answer = cls._extract_answer(sample.preds[0])
            if answer is None:
                fail_cnt += 1
                logger.warning(f"Failed to extract answer from prediction: {sample.preds[0]}")
                continue
            answer = answer[0]
            assert answer in 'abcdefg', f"Invalid answer {answer} parsed from '{sample.preds[0]}'"
            answer = 'abcdefg'.index(answer)

            if sample.item['error_type'] == 'ok' or sample.item['error_type'] == 'expert':
                gt = sample.item['answer']
            elif sample.item['error_type'] == 'wrong_groundtruth':
                gt = sample.item['correct_answer']
            else:
                # Other kinds of errors
                num_samples -= 1
                continue

            if gt == answer:
                correct_cnt += 1
            else:
                incorrect_cnt += 1
                if incorrect_cnt % 50 == 0:
                    logger.warning(f"Incorrect answer: {sample.preds[0]} (expected: {gt}, got: {answer})")
        accuracy = correct_cnt / num_samples if num_samples > 0 else 0.0
        logger.info(f"Failed to extract answer from {fail_cnt} predictions out of {len(predictions)}")
        logger.info(f"Accuracy: {accuracy:.4f} ({correct_cnt}/{num_samples})")
        return {"accuracy": accuracy, "correct": correct_cnt, "total": num_samples}

    @classmethod
    def _extract_answer(cls, pred: str) -> str:
        _pred = pred.strip().lower()  # Normalize whitespace
        # Pattern-1: A. XXX
        match = re.search(r"^([a-d])\.\s+.*", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-2: answer: D
        match = re.search(r"^answer:\s*([a-d])\s*$", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-3: search for "answer: D"
        match = re.search(r"answer:\s*([a-d])", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-4: search for "**answer**: B"
        match = re.search(r"\*\*answer\*\*:\s*([a-d])", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-5: search for "**D. 0.45**"
        match = re.search(r"\*\*([a-d])\.\s*.*?\*\*", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-6: search for "**answer**: B. xxx"
        match = re.search(r"\*\*answer:\*\*\s*([a-d])\.\s*.*?", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-6: search for "**answer:**: B"
        match = re.search(r"\*\*answer:\*\*\s*([a-d])", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-7: B
        match = re.match(r"^[a-d]\s*$", _pred)
        if match:
            return match.group(0).strip()
        # Pattern-8: \boxed{a}
        match = re.search(r"\\boxed{([a-d])}", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-9: \boxed{\text{a}}
        match = re.search(r"\\boxed{\s*\\text{([a-d].*?)}\s*}", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-10: Answer: **B**
        match = re.search(r"answer:\s*\*\*([a-d])\*\*", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-11: **answer**: **B**
        match = re.search(r"\*\*answer\*\*:\s*\*\*([a-d])\*\*", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-12: \text{answer: } B
        match = re.search(r"\\text{answer:\s*}\s*([a-d])", _pred)
        if match:
            return match.group(1).strip()
        # Pattern-13: **Answer**: "D"
        match = re.search(r"\*\*answer\*\*:\s*\"([a-d])\"", _pred)
        if match:
            return match.group(1).strip()
        # Pattern: \text{answer}: \text{D}
        match = re.search(r"\\text{answer}:\s*\\text{([a-d])}", _pred)
        if match:
            return match.group(1).strip()
        # Pattern: \text{answer}: "C"
        match = re.search(r'\\text{answer}:\s*"([a-d])"', _pred)
        if match:
            return match.group(1).strip()
        # Pattern: \boxed{\text{C. } 4}
        match = re.search(r"\\boxed{\\text{([a-d].*?)}.*}", _pred)
        if match:
            return match.group(1).strip()
        return None
