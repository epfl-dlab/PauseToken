from datasets import load_from_disk
import os
import re
import argparse
import json
DATA_DIR = "../data/" 

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


def inject_pause_to_str(input_string, n_pauses_per_patterns, pause_token):
    """ Inject pauses in a string based on the patterns provided
    
    :param input_string: The string in which the pauses are to be injected
    :type input_string: str
    :param n_pauses_per_patterns: A dictionary where the key is the pattern and the value is the number of pauses to be injected after the pattern
    :type n_pauses_per_patterns: dict
    :param pause_token: The pause token to be injected
    :type pause_token: str
    :return: The string with the pauses injected
    :rtype: str
    """
    patterns = list(n_pauses_per_patterns.keys())
    
    pattern_occurences = []
    for pat in patterns:
        pattern_occurences.extend(find_pattern(input_string, pattern= pat))
    
   
    pattern_occurences.sort(key=lambda x: x["start"], reverse=True)
    augmented_string = input_string
    for patt in pattern_occurences:
        augmented_string =  add_pause(
            string = augmented_string,
            idx = patt["start"] + 1,
            n_pauses = n_pauses_per_patterns[patt["pattern"]],
            pause_token = pause_token
        )
    
    # Add pauses at the beginning of the string if one of the patterns is \n
    if "\n" in patterns:
        
        augmented_string =  add_pause(
                string = augmented_string,
                idx = 0,
                n_pauses = n_pauses_per_patterns["\n"],
                pause_token = pause_token
            )
    return augmented_string


def inject_pauses(
        sample,
        n_pauses_per_patterns = {
            r"=": 1,
            r"\n": 1
            },
        pause_token = "<|PAUSE|>",
        pause_augm_col_name = "pause_augmented_answer",
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
    sample[pause_augm_col_name]  = inject_pause_to_str(input_string, n_pauses_per_patterns, pause_token)
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
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Summary of Parameters:")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
    
    return args


if __name__ == "__main__":
    
    args = parse_args()
    
    if args.verbose:
        print("Loading Dataset...")
    dataset = load_from_disk(args.dataset_location)
    
    if args.verbose:
        print("Injecting Pauses...")
    
    dataset = dataset.map(
        lambda sample: inject_pauses(sample,args.n_pauses_per_patterns, args.pause_token)
    )
    
    if args.verbose:
        print("done !")
        print("viewing a sample from training data: ")
        print(dataset["train"][0])
        
        print("Saving Augmented Dataset...")
    
    dataset.save_to_disk(args.augm_dataset_save_location)