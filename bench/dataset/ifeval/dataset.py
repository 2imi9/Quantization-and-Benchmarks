from loguru import logger
from datasets import load_dataset, Dataset

from ..base import BaseDataset
from . import evaluation_lib


class IFEvalDataset(BaseDataset):

    DATASET_NAME = "google/IFEval"

    @classmethod
    def load(cls, debug: bool = False) -> Dataset:
        ds: Dataset = load_dataset(cls.DATASET_NAME, split="train")
        if debug:
            ds = ds.select(range(10))
        return ds

    @classmethod
    def _measure_sample(cls, sample) -> tuple:
        response = sample.preds[0]
        doc = sample.item
        inp = evaluation_lib.InputExample(
            key=doc["key"],
            instruction_id_list=doc["instruction_id_list"],
            prompt=doc["prompt"],
            kwargs=[{k: v for k, v in kwargs.items() if v is not None} for kwargs in doc['kwargs']],
        )
        out_strict = evaluation_lib.test_instruction_following_strict(inp, response)
        out_loose = evaluation_lib.test_instruction_following_loose(inp, response)
        return out_strict, out_loose

    @classmethod
    def measure(cls, predictions: list) -> dict:
        stricts, looses = list(), list()
        for sample in predictions:
            out_strict, out_loose = cls._measure_sample(sample)
            stricts.append(out_strict)
            looses.append(out_loose)

        stricts_report = evaluation_lib.get_report(stricts)
        looses_report = evaluation_lib.get_report(looses)
        logger.info(f"Strict: {stricts_report}")
        logger.info(f"Loose: {looses_report}")
        return {
            "strict": stricts_report,
            "loose": looses_report,
        }
