from loguru import logger
from datasets import Dataset, load_dataset, concatenate_datasets

from ..base import BaseDataset
from .evaluate import postprocess_pred, string_match_part, string_match_all

# https://github.com/open-compass/opencompass/blob/main/examples/eval_ruler.py
# https://github.com/NVIDIA/RULER/blob/main/scripts/data/prepare.py For data generation
# Test on 32K https://huggingface.co/datasets/lighteval/RULER-32768-Qwen-3


class RulerNIAHDataset(BaseDataset):

    TASK_NAME = "RULER-NIAH-32k"
    DATASET_NAME = "lighteval/RULER-32768-Qwen-3"
    CONFIGS = ["niah_single_1", "niah_multikey_1", "niah_multivalue", "niah_multikey_3", "niah_single_3", "niah_single_2", "niah_multikey_2", "niah_multiquery"]
    METRIC_FN = string_match_all

    @classmethod
    def load(cls, debug: bool = False) -> Dataset:
        ds = load_dataset(cls.DATASET_NAME)
        configs = cls.CONFIGS[:1] if debug else cls.CONFIGS
        logger.info(f"Loading dataset {cls.DATASET_NAME} with configs: {configs}")
        datasets = [ds[conf] for conf in configs]
        ds = concatenate_datasets(datasets)
        if debug:
            ds = ds.select(range(10))
        return ds

    @classmethod
    def measure(cls, predictions: list) -> dict:

        inputs, predicts = list(), list()
        references, indices = list(), list()
        for sample in predictions:
            item = sample.item
            pred = postprocess_pred(sample.preds[0])
            reference = item['outputs']
            index = item['index']
            inputs.append(item['input'])
            predicts.append(pred)
            references.append(reference)
            indices.append(index)

        # Calculate the score
        task_nulls = f'{sum([len(x)==0 for x in predicts])}/{len(predicts)}'
        task_score = cls.METRIC_FN(predicts, references)
        metrics = {
            "task_name": cls.TASK_NAME,
            "task_score": task_score,
            "task_nulls": task_nulls,
        }
        logger.info(f"Task: {cls.TASK_NAME}, Score: {task_score}, Nulls: {task_nulls}")
        return metrics


class RulerNIAHSingle1Dataset(RulerNIAHDataset):
    # NIAH Single Key

    CONFIGS = ["niah_single_1"]
    TASK_NAME = "RULER-NIAH-SINGLE-1-32k"


class RulerNIAHSingle2Dataset(RulerNIAHDataset):
    # NIAH Single Key 2

    CONFIGS = ["niah_single_2"]
    TASK_NAME = "RULER-NIAH-SINGLE-2-32k"


class RulerNIAHSingle3Dataset(RulerNIAHDataset):
    # NIAH Single Key 3

    CONFIGS = ["niah_single_3"]
    TASK_NAME = "RULER-NIAH-SINGLE-3-32k"


class RulerNIAHMultiKey1Dataset(RulerNIAHDataset):
    # NIAH Multi Key 1

    CONFIGS = ["niah_multikey_1"]
    TASK_NAME = "RULER-NIAH-MULTIKEY-1-32k"


class RulerNIAHMultiKey2Dataset(RulerNIAHDataset):
    # NIAH Multi Key 2

    CONFIGS = ["niah_multikey_2"]
    TASK_NAME = "RULER-NIAH-MULTIKEY-2-32k"


class RulerNIAHMultiKey3Dataset(RulerNIAHDataset):
    # NIAH Multi Key 3

    CONFIGS = ["niah_multikey_3"]
    TASK_NAME = "RULER-NIAH-MULTIKEY-3-32k"


class RulerNIAHMultiValueDataset(RulerNIAHDataset):
    # NIAH Multi Value

    CONFIGS = ["niah_multivalue"]
    TASK_NAME = "RULER-NIAH-MULTIVALUE-32k"


class RulerNIAHMultiQueryDataset(RulerNIAHDataset):
    # NIAH Multi Query

    CONFIGS = ["niah_multiquery"]
    TASK_NAME = "RULER-NIAH-MULTIQUERY-32k"


class RulerCWEDatset(RulerNIAHDataset):
    # Common Words Extraction

    CONFIGS = ["cwe"]
    TASK_NAME = "RULER-CWE-32k"


class RulerFWEDataset(RulerNIAHDataset):
    # Frequent Words Extraction

    CONFIGS = ["fwe"]
    TASK_NAME = "RULER-FWE-32k"


class RulerVTDataset(RulerNIAHDataset):
    # Variable Tracking

    CONFIGS = ["vt"]
    TASK_NAME = "RULER-VT-32k"
