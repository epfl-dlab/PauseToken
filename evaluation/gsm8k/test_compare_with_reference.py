from evaluation.gsm8k.compare_with_reference import judge
from thirdparty.openai.grade_school_math.dataset import read_jsonl

train_path = "data/gsm8k/train.json"
train_examples = read_jsonl(train_path)

corrupted_examples = train_examples[:3]

def test_judge_all_correct():
    results = judge(train_examples, train_examples)
    assert all(results)