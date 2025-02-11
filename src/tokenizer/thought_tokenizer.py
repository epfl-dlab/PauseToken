from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from typing import List, Union
import numpy as np
import torch
import re
def create_thought_tokenizer(tokenizer, start_tag: str = "<th>", end_tag: str = "</th>"):
    """Dynamically create a subclass of the tokenizer and return an instance."""
    class ThoughtTokenizerWrapper(tokenizer.__class__):
        
        def __init__(self, *args,  start_tag: str = "<|", end_tag: str = "|>", **kwargs):
            # Call the parent's __init__ with the tokenizer's config
            super().__init__(*args, **kwargs)
            self.start_tag = start_tag
            self.end_tag = end_tag

        def decode(self, token_ids, *args, **kwargs):
            """Override decode while calling the original method."""
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.cpu().numpy()
            elif isinstance(token_ids, List):
                token_ids = np.array(token_ids)

            thought_mask = (token_ids >= self.vocab_size)
            thought_indices = np.where(thought_mask)[0]
            
            text_components = []
            for i, thought_index in enumerate(thought_indices):
                if i == 0 and thought_index > 0:
                    text_components.append(super().decode(token_ids[:thought_index], *args, **kwargs))
                elif ((thought_indices[i-1] +1) - thought_index) > 0:
                    text_components.append(super().decode(token_ids[thought_indices[i-1] + 1:thought_index], *args, **kwargs))
         
                thought_token = super().decode(token_ids[thought_index:thought_index+1] - self.vocab_size, *args, **kwargs)
        
                text_components.append(f"{self.start_tag}{thought_token}{self.end_tag}")

            if thought_indices[-1] < len(token_ids):
                text_components.append(super().decode(token_ids[thought_indices[-1] + 1:], *args, **kwargs))

            final_text = "".join(text_components)
            
            return final_text

        def batch_decode(self, sequences, *args, **kwargs):
            """Override batch_decode while calling the original method."""
            if isinstance(sequences, torch.Tensor):
                sequences = sequences.cpu().numpy()
            elif isinstance(sequences, List):
                sequences = np.array(sequences)
            if len(sequences.shape) == 1:
                sequences = sequences.reshape(-1, 1)

            return [self.decode(sequence, *args, **kwargs) for sequence in sequences]
        
        def find_tag_positions(self, text: str) -> list:
            """Find all occurrences of text enclosed between start_tag and end_tag, including start & end indices."""
            pattern = re.escape(self.start_tag) + r".*?" + re.escape(self.end_tag)  # Non-greedy match
            matches = [(match.start(), match.end(), match.group()) for match in re.finditer(pattern, text, re.DOTALL)]
            return matches                                        

        def __call__(self, text, **kwargs):
            """Override __call__ while calling the original method."""
            # typing.Union[str, typing.List[str], typing.List[typing.List[str]]] = None
            if isinstance(text, str):
                all_matches = [self.find_tag_positions(text)]
                all_texts = [text]
            elif isinstance(text, List) and isinstance(text[0], str):
                all_matches = [self.find_tag_positions(text) for text in text]
                all_texts = text
            else:
                raise ValueError(f"Input must be a string or a list of strings. you gave: {text}. Consider refactoring this method if you wish to pass typing.List[typing.List[str]]]")

            last_end = 0
            padding = kwargs.get("padding", False)
            all_tokenized_texts = []
            all_attention_masks = []
            for matches,all_text in zip(all_matches, all_texts):
                ls_tokenized_text = []
                ls_attention_mask = []
                for match in matches:
                    start = match[0]
                    end = match[1]
                    word = match[2]
                    word = word.replace(self.start_tag, "").replace(self.end_tag, "")
                    if len(all_text[last_end:start]) > 0:
                        tokenized_output = super().__call__(all_text[last_end:start], padding=False)
                        ls_tokenized_text.extend(tokenized_output["input_ids"])
                        ls_attention_mask.extend(tokenized_output["attention_mask"])
                    tokenized_word_output = super().__call__(word, padding=False)
                    assert len(tokenized_word_output["input_ids"]) == 1, f"A thought can only be associated to a single token"
                    ls_tokenized_text.append(tokenized_word_output["input_ids"][0] + self.vocab_size)
                    ls_attention_mask.append(tokenized_word_output["attention_mask"][0])
                    last_end = end
                if last_end < len(all_text):
                    tokenized_output = super().__call__(all_text[last_end:], padding=False)
                    ls_tokenized_text.extend(tokenized_output["input_ids"])
                    ls_attention_mask.extend(tokenized_output["attention_mask"])

                all_tokenized_texts.append(ls_tokenized_text)

            if padding:
                pad_kwargs = {
                    "padding": padding,
                    "max_length": kwargs.get("max_length", None),
                    "pad_to_multiple_of": kwargs.get("pad_to_multiple_of", None),
                    "padding_side": kwargs.get("padding_side", None),
                    "return_attention_mask": kwargs.get("return_attention_mask", None),
                    "return_tensors": kwargs.get("return_tensors", None),
                    "verbose": kwargs.get("verbose", True),
                }
                result = self.pad({"input_ids": all_tokenized_texts}, **pad_kwargs)

            else:
                result = {"input_ids": all_tokenized_texts, "attention_mask": all_attention_masks}

            return result
                
    return ThoughtTokenizerWrapper(start_tag=start_tag, end_tag=end_tag, **tokenizer.init_kwargs, )


def test_tokenizer():
    from transformers import AutoTokenizer
    
    tokenizer = create_thought_tokenizer(AutoTokenizer.from_pretrained("gpt2"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    test = "This is <th>a</th> tokenization test <th>!</th><th>!</th><th>!</th>."
    result_text_to_tok = tokenizer(test, return_tensors="pt", padding = True)
    result_tok_to_text = tokenizer.batch_decode(result_text_to_tok["input_ids"])
    print("Exp1: ")
    print(" Original: ", test)
    print(" text -> tok", result_text_to_tok["input_ids"])
    print(" mask", (result_text_to_tok["input_ids"] >= tokenizer.vocab_size))
    print(" tok --> text: ", result_tok_to_text)


    test = ["This is <th>a</th> tokenization test <th>!</th><th>!</th><th>!</th>>.", "<th>!</th><th>!</th><th>!</th> This is <th>a</th> tokenization test boooo."]
    result_text_to_tok = tokenizer(test, return_tensors="pt", padding = True)
    result_tok_to_text = tokenizer.batch_decode(result_text_to_tok["input_ids"])
    print("Exp1: ")
    print(" Original: ", test)
    print(" text -> tok", result_text_to_tok["input_ids"])
    print(" mask", (result_text_to_tok["input_ids"] >= tokenizer.vocab_size))
    print(" tok --> text: ", result_tok_to_text)

if __name__ == "__main__":
    test_tokenizer()