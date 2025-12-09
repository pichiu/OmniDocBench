from .detection_eval import DetectionEval
from .end2end_run_eval import End2EndEval
from .recognition_eval import RecognitionBaseEval
from .ocr_comparison_eval import OCRComparisonEval

from registry.registry import EVAL_TASK_REGISTRY

__all__ = [
    "RecognitionBaseEval",
    "DetectionEval",
    "End2EndEval",
    "OCRComparisonEval",
]

print("EVAL_TASK_REGISTRY: ", EVAL_TASK_REGISTRY.list_items())