from evaluation.gsm8k.compare_with_reference import judge
from thirdparty.openai.grade_school_math.dataset import read_jsonl
from copy import deepcopy

import pytest

train_path = "data/gsm8k/train.json"
train_examples = read_jsonl(train_path)

def test_judge_all_correct():
    results = judge(train_examples, train_examples)
    assert all(results)

def test_judge_some_incorrect():
    corrupted_examples = deepcopy(train_examples[:3])
    for corrupted_example in corrupted_examples:
        corrupted_example["answer"] = "#### -100000"
    results = judge(corrupted_examples, train_examples[:3])
    assert not all(results)

if __name__=="__main__":
    pytest.main([__file__])
