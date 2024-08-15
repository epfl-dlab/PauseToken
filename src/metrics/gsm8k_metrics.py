from src.utils.trainer_utils import extract_answer, INVALID_ANS

def is_correct(model_completion: str, gt_example: str) -> bool:
    """ Check if the model completion is correct given the ground truth example. Completions must be in the GSM8K dataset format
    
    :param model_completion: Model completion
    :type model_completion: str
    
    """
    gt_answer = extract_answer(gt_example)
    assert gt_answer != INVALID_ANS, \
        f"Ground truth answer is invalid and doesn't follow the GSM8K formate, your ground truth answer is {gt_example['answer']}"
    return extract_answer(model_completion) == gt_answer