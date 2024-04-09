from thirdparty.openai.grade_school_math.dataset import is_correct
from typing import List, Dict

def judge(model_outputs: List[str], ground_truths:List[str]):

    assert len(model_outputs) == len(ground_truths)
    results = []
    for model_output, ground_truth in zip(model_outputs, ground_truths):
        results.append(is_correct(model_output, ground_truth))

    return results
