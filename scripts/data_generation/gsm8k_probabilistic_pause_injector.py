from datasets import load_from_disk,load_dataset,concatenate_datasets
import os
import re
import argparse
import json
import random
from transformers import AutoTokenizer
DATA_DIR = "data/" 


def dict_type(string):
    """ Convert a string to a dictionary
    
    :param string: A string that represents a dictionary
    :type string: str
    :return: A dictionary
    :rtype: dict
    """
    try:
        return json.loads(string)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid dictionary format. Must be a valid JSON string.")
    

def find_pattern(input_string,pattern):
    """ Find all occurences of a pattern in a string
    
    :param input_string: The string in which the pattern is to be found
    :type input_string: str
    :param pattern: The pattern to be found
    :type pattern: str
    :return: A list of dictionaries containing the start and end indices of the pattern in the string and the pattern itself
    :rtype: list[dict]
    """
    # Use regular expression to find all matches
    matches = re.finditer(pattern, input_string)
    
    # Store indices of matches in a list
    equal_sign_indices = [{"start": match.start(), "end": match.end(), "pattern": pattern} for match in matches]
    
    return equal_sign_indices

def add_pause(string, idx ,n_pauses, pause_token):
    """ Add n_pauses number of pause tokens at a specific index in a string
    
    :param string: The string in which the pause tokens are to be added
    :type string: str
    :param idx: The index at which the pause tokens are to be added
    :type idx: int
    :param n_pauses: The number of pause tokens to be added
    :type n_pauses: int
    :param pause_token: The pause token to be added
    :type pause_token: str
    :return: The string with the pause tokens added
    :rtype: str
    """
    pause_toks = n_pauses * pause_token
    return string[:idx] + pause_toks + string[idx:]


def inject_pause_to_str(input_string, pause_prob, pause_token, tokenizer):
    """ Inject pauses in a string based on the patterns provided
    
    :param input_string: The string in which the pauses are to be injected
    :type input_string: str
    :param pause_prob: Probability of injecting a pause token after each token in the text
    :type pause_prob: float
    :param pause_token: The pause token to be injected
    :type pause_token: str
    """

    augmented_string = input_string

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided to inject random pauses")
    else:
        tokenizer.add_tokens([pause_token], special_tokens=True)
        splited_augm_str = tokenizer(augmented_string)["input_ids"]
        if tokenizer.bos_token_id is not None and splited_augm_str[0] == tokenizer.bos_token_id:
            splited_augm_str = splited_augm_str[1:]
            
        pause_token = tokenizer.convert_tokens_to_ids(pause_token)

    
    length_of_tokens = len(splited_augm_str)
    pause_location = random.choices([0,1], weights=[1-pause_prob, pause_prob], k=length_of_tokens)
    
    for inv_idx, do_pause in enumerate(pause_location[::-1]):
        if do_pause == 1:
            idx = length_of_tokens - inv_idx - 1
            splited_augm_str.insert(idx, pause_token)
        
    augmented_string = tokenizer.decode(splited_augm_str)
    
    return augmented_string


def inject_pauses(
        sample,
        pause_prob,
        pause_token,
        pause_augm_col_name,
        tokenizer,
    ):
    """ function used in map to inject pauses in a sample
    
    :param sample: The sample in which the pauses are to be injected
    :type sample: dict
    :param pause_prob: Probability of injecting a pause token after each token in the text
    :type pause_prob: float
    :param pause_token: The pause token to be injected
    :type pause_token: str
    :param pause_augm_col_name: The name of the column where the augmented text will be stored
    :type pause_augm_col_name: str
    :param tokenizer: The tokenizer to be used to insert random pauses
    :type tokenizer: transformers.AutoTokenizer
    """
   
    input_string = sample["answer"]
    sample[pause_augm_col_name]  = inject_pause_to_str(
        input_string,
        pause_prob=pause_prob,
        pause_token=pause_token,
        tokenizer=tokenizer
    )
    return sample


def parse_args():
    """ Parse the arguments for the pause injection script"""
    parser = argparse.ArgumentParser(description="Arg Parser for pause injection")
    
    parser.add_argument(
        "--dataset_location",
        type=str,
        default= os.path.join(DATA_DIR, "gsm8k"),
        help="Specify the location of the dataset"
    )
    parser.add_argument(
        "--pause_token",
        type=str,
        default="<|pause|>",
        help="The pause token string to be injected"
    )
    
    parser.add_argument(
        "--pause_probability",
        default=0.1,
        type=float,
        help="Probability of injecting a pause token after each token in the text"
    )
    
    parser.add_argument(
        "--tokenizer_hf_name",
        type=str,
        default="None",
        help="The name of the Hugging Face tokenizer to be used to insert random pauses. \
        If None, spaces ' ' will be used to insert random pauses."
    )
    
    parser.add_argument(
        "--augm_dataset_save_location",
        type=str,
        default=os.path.join(DATA_DIR, "gsm8k_pause_injected"),
        help="Specify the location of the output dataset"
    )
    
    parser.add_argument(
        "--pause_augm_col_name",
        type=str,
        default="pause_augmented_answer",
        help="Specify the name of the column where the augmented text will be stored"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random number generation"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Summary of Parameters:")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
    
    return args


if __name__ == "__main__":
    
    args = parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    if args.verbose:
        print("Loading Dataset...")
    
    data_files = {"train": os.path.join(args.dataset_location, "train.json"),
                  "test": os.path.join(args.dataset_location, "test.json")}
    
    
    train_dataset = load_dataset("json", data_files=data_files, split="train")

    test_dataset = load_dataset("json", data_files=data_files, split="test")

    splits = ["train", "val", "test"]
    for split, dataset in zip(splits, [train_dataset, test_dataset]):
        if args.tokenizer_hf_name:
            if args.verbose:
                print("Loading Tokenizer ...")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_hf_name)
        else:
            if args.verbose:
                print("No Tokenizer provided. Using spaces to insert random pauses.")
            tokenizer = None
        
        if args.verbose:
            print("Injecting Pauses...")
        
        augmented_ds = []
        if split != "test":
            dataset =  dataset.map(
                    lambda sample: inject_pauses(
                        sample,
                        pause_prob=args.pause_probability,
                        pause_token=args.pause_token,
                        pause_augm_col_name=args.pause_augm_col_name,
                        tokenizer=tokenizer
                    ),
                    load_from_cache_file=False
                )
        
        if args.verbose:
            print("done !")
            print("viewing a sample from training data: ")
            print(dataset[0])
            
            print("Saving Augmented Dataset...")
        dataset.to_json(os.path.join(args.augm_dataset_save_location,f"{split}.json"),batch_size= len(dataset), indent = 2, lines=False)
