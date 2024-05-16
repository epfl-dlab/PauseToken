from dataclasses import dataclass
from trl import DataCollatorForCompletionOnlyLM
from collections.abc import Mapping
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
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

