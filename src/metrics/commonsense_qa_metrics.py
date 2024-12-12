from src.utils.trainer_utils import extract_answer, INVALID_ANS

def is_correct(model_completion: str, gt_example: str) -> bool:
    """ Check if the model completion is correct given the ground truth example. Completions must be in the GSM8K dataset format
    
    :param model_completion: Model completion
    :type model_completion: str
    
    """
    #print(gt_example)
    #print(gt_example.split("Answer: "))
    #print("------------------------------")
    #print(model_completion)
    gt_answer = gt_example[0]
    if gt_answer not in ["A", "B", "C", "D", "E"]:
        f"Ground truth answer is invalid and doesn't follow the GSM8K formate, your ground truth answer is {gt_example}"
    try:
        print("eval:", model_completion.split("Answer:")[1][0], gt_answer, model_completion.split("Answer:")[1][0] == gt_answer)
        return model_completion.split("Answer:")[1][0] == gt_answer
    except:
        return False