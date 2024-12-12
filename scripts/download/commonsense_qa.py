import pandas as pd
import os

splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/tau/commonsense_qa/" + splits["train"])
df_valid = pd.read_parquet("hf://datasets/tau/commonsense_qa/" + splits["validation"])
df_test = pd.read_parquet("hf://datasets/tau/commonsense_qa/" + splits["test"])

def convert_df_to_qstr_astr(d):
    #print(d)
    qstr = "Answer the following question\n"
    qstr += d['question']
    qstr += "\nChoices:\n"
    for l, t in zip(d['choices']['label'], d['choices']['text']):
        qstr += f"{l}: {t}\n"
    astr = d['answerKey']
    return pd.Series({"question" : qstr, "answer": astr})

rewritten_train = df_train.apply(convert_df_to_qstr_astr, axis=1)
rewritten_valid = df_valid.apply(convert_df_to_qstr_astr, axis=1)
rewritten_test = df_test.apply(convert_df_to_qstr_astr, axis=1)

# export as jsonl
os.makedirs("./data/commonsense_qa", exist_ok=True)
rewritten_train.to_json("./data/commonsense_qa/train.json", orient='records', lines=True)
rewritten_valid.to_json("./data/commonsense_qa/test.json", orient='records', lines=True)
#rewritten_test.to_json("../data/commonsense_qa/test.json", orient='records', lines=True)