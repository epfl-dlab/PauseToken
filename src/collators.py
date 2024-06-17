from dataclasses import dataclass
from trl import DataCollatorForCompletionOnlyLM
from transformers.data.data_collator import DataCollatorMixin
from collections.abc import Mapping
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import torch
@dataclass
class DataCollatorForCompletionOnlyLMWSFT(DataCollatorForCompletionOnlyLM):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        #unpack nested examples
        unpacked_examples = []
        if isinstance(examples[0]["input_ids"], list):
            unpacked_examples = []
            for example in examples:
                for values in zip(*example.values()):
                    unpacked_examples.append({key: value for key, value in zip(example.keys(), values)})
        else:
            unpacked_examples = examples
        
        batch = super().torch_call(unpacked_examples)
        return batch
    
@dataclass
class ConditionalCollator(DataCollatorMixin):
    return_tensors: str = "pt"
    def __init__(self, name_to_collator: Dict[int, DataCollatorMixin], name_col:str,*args, **kwargs):
        self.name_to_collator = name_to_collator
        self.name_col = name_col
        super().__init__(*args, **kwargs)
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        train_method = examples[0][self.name_col]
        collator = self.name_to_collator[train_method]
        data = [example["data"] for example in examples]
        tensor = collator(data)
        tensor[self.name_col] = torch.tensor([example[self.name_col] for example in examples]).long()
        return tensor