
from datasets import Dataset, load_dataset
import os

def DatasetFromFile(path, **kwargs) -> Dataset:

    file_type = kwargs.get("file_type")
    files = kwargs.get("files")
    data_files={file: os.path.join(path, file)+"."+file_type for file in files}
    
    dataset = load_dataset(file_type, data_files=data_files,)
    if kwargs.get('input_label') and kwargs.get('output_label'):
        dataset = dataset.rename_column(kwargs.get('input_label'), "input").rename_column(kwargs.get('output_label'), "output")

    if not 'val' in dataset.keys():
        val_size = kwargs.get('train_val_test_split', [-1, 0.1])[1]
        data_train = dataset['train'].train_test_split(test_size=val_size)
        dataset['train'] = data_train['train']  
        dataset['val'] = data_train['test']

    debug_n = kwargs.get('debug_n', None)
    if debug_n is not None:
        dataset["train"] = dataset["train"].select(range(debug_n))
        dataset["val"] = dataset["val"].select(range(debug_n))
        dataset["test"] = dataset["test"].select(range(debug_n)) if "test" in dataset.keys() else None

    return dataset