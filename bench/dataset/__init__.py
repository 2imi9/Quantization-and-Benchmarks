from .ifeval import IFEvalDataset
from .math500 import Math500Dataset
from .mmlu_redux import MmluReduxDataset
from .livecodebench import LiveCodeBenchV5Dataset
from .ruler import RulerCWEDatset, RulerFWEDataset, RulerNIAHDataset, RulerVTDataset, \
    RulerNIAHMultiKey1Dataset, RulerNIAHMultiKey2Dataset, RulerNIAHMultiKey3Dataset, \
    RulerNIAHMultiQueryDataset, RulerNIAHMultiValueDataset, RulerNIAHSingle3Dataset, \
    RulerNIAHSingle2Dataset, RulerNIAHSingle1Dataset
from .aime25 import AIME25Dataset


DATASETS = {
    "mmlu-redux": MmluReduxDataset,
    "ifeval": IFEvalDataset,
    "math500": Math500Dataset,
    "livecode_v5": LiveCodeBenchV5Dataset,
    # Ruler
    "ruler-niah-32k": RulerNIAHDataset,
    "ruler-cwe-32k": RulerCWEDatset,
    "ruler-fwe-32k": RulerFWEDataset,
    "ruler-vt-32k": RulerVTDataset,
    "ruler-niah-multikey-1-32k": RulerNIAHMultiKey1Dataset,
    "ruler-niah-multikey-2-32k": RulerNIAHMultiKey2Dataset,
    "ruler-niah-multikey-3-32k": RulerNIAHMultiKey3Dataset,
    "ruler-niah-multiquery-32k": RulerNIAHMultiQueryDataset,
    "ruler-niah-multivalue-32k": RulerNIAHMultiValueDataset,
    "ruler-niah-single-1-32k": RulerNIAHSingle1Dataset,
    "ruler-niah-single-2-32k": RulerNIAHSingle2Dataset,
    "ruler-niah-single-3-32k": RulerNIAHSingle3Dataset,
    "aime25": AIME25Dataset,
}
