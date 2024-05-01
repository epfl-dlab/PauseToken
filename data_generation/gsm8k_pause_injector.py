from datasets import load_from_disk,load_dataset
import os
import re
import argparse
import json
import random
from transformers import AutoTokenizer
import sys
sys.path.append("..")
from src.utils import dict_type
DATA_DIR = "../data/" 



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


def inject_pause_to_str(input_string, n_pauses_per_patterns, pause_token,n_random_pauses, tokenizer):
    """ Inject pauses in a string based on the patterns provided
    
    :param input_string: The string in which the pauses are to be injected
    :type input_string: str
    :param n_pauses_per_patterns: A dictionary where the key is the pattern and the value is the number of pauses to be injected after the pattern
    :type n_pauses_per_patterns: dict
    :param pause_token: The pause token to be injected
    :type pause_token: str
    :param n_random_pauses: The number of random pauses to be injected (using uniform distribution)
    :type n_random_pauses: int
    :return: The string with the pauses injected
    :rtype: str
    """
    patterns = list(n_pauses_per_patterns.keys())
    
    pattern_occurences = []
    for pat in patterns:
        res = find_pattern(input_string, pattern= pat) 
        pattern_occurences.extend(res)
   
    pattern_occurences.sort(key=lambda x: x["start"], reverse=True)
    augmented_string = input_string
    for patt in pattern_occurences:
        augmented_string =  add_pause(
            string = augmented_string,
            idx = patt["end"],
            n_pauses = n_pauses_per_patterns[patt["pattern"]],
            pause_token = pause_token
        )
    
    # Add pauses at the beginning of the string if one of the patterns is \n
    if "\n" in patterns and n_pauses_per_patterns["\n"] > 0:
        augmented_string =  add_pause(
                string = augmented_string,
                idx = 0,
                n_pauses = n_pauses_per_patterns["\n"],
                pause_token = pause_token
            )
        
    if n_random_pauses > 0:
        if tokenizer is None:
            splited_augm_str = augmented_string.split(" ")
        else:
            tokenizer.add_tokens([pause_token], special_tokens=True)
            splited_augm_str = tokenizer(augmented_string)["input_ids"]
            pause_token = tokenizer(pause_token)["input_ids"][1]
        random_indices = [random.randint(0, len(splited_augm_str)) for _ in range(n_random_pauses)]
        random_indices.sort(reverse=True)
        for idx in random_indices:
            splited_augm_str.insert(idx, pause_token)
        if tokenizer is None:
            augmented_string = " ".join(splited_augm_str)
        else:
            augmented_string = tokenizer.decode(splited_augm_str)
            
    return augmented_string


def inject_pauses(
        sample,
        n_pauses_per_patterns = {
            r"=": 1,
            r"\n": 1
            },
        n_random_pauses=0,
        pause_token = "<|PAUSE|>",
        pause_augm_col_name = "pause_augmented_answer",
        tokenizer = None,
    ):
    """ function used in map to inject pauses in a sample
    
    :param sample: The sample in which the pauses are to be injected
    :type sample: dict
    :param n_pauses_per_patterns: A dictionary where the key is the pattern and the value is the number of pauses to be injected after the pattern
    :type n_pauses_per_patterns: dict
    :param pause_token: The pause token to be injected
    :type pause_token: str
    """
    
    input_string = sample["answer"]
    sample[pause_augm_col_name]  = inject_pause_to_str(input_string, n_pauses_per_patterns, pause_token,n_random_pauses, tokenizer)
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
        default="<|PAUSE|>",
        help="The pause token string to be injected"
    )
    
    parser.add_argument(
        "--n_pauses_per_patterns",
        type=dict_type,
        default={
            r"=": 1,
            r"\n": 1
        },
        help="A dictionary of key value pairs where key \
            is the pattern and value is the number of pauses \
                to be injected after an occurence of that pattern"
    )
    
    parser.add_argument(
        "--n_random_pauses",
        default=0,
        type=int,
        help="The number of pauses to be injected at random locations (using uniform distribution)"
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
        default=os.path.join(DATA_DIR, "gs8k_pause_injected"),
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
        default=123,
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
    dataset = load_dataset("json", data_files=data_files)
    
    
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
    dataset = dataset.map(
        lambda sample: inject_pauses(sample,args.n_pauses_per_patterns,args.n_random_pauses ,args.pause_token, args.pause_augm_col_name, tokenizer)
    )
    
    if args.verbose:
        print("done !")
        print("viewing a sample from training data: ")
        print(dataset["train"][0])
        
        print("Saving Augmented Dataset...")
    dataset["train"].to_json(os.path.join(args.augm_dataset_save_location,"train.json"))
    dataset["test"].to_json(os.path.join(args.augm_dataset_save_location,"test.json"))