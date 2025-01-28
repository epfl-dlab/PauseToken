
from datasets import Dataset, load_dataset
import os
import hydra
def DatasetFromFile(path, **kwargs) -> Dataset:

    file_type = kwargs.get("file_type")
    files = kwargs.get("files")
    data_files={file: os.path.join(path, file)+"."+file_type for file in files}
    dataset = load_dataset(file_type, data_files=data_files,)
    if kwargs.get('input_label') and kwargs.get('output_label'):
        dataset = dataset.rename_column(kwargs.get('input_label'), "input").rename_column(kwargs.get('output_label'), "output")

    #additional_transformations
    additional_transformation = kwargs.get("additional_transformation", None)
    if additional_transformation:
        if not callable(additional_transformation):
            additional_transformation = hydra.utils.instantiate(additional_transformation)
        dataset = dataset.map(additional_transformation, batched=True,load_from_cache_file=False)

    if not 'val' in dataset.keys() and "train" in dataset.keys():
        val_size = kwargs.get('train_val_test_split', [-1, 0.1])[1]
        data_train = dataset['train'].train_test_split(test_size=val_size, seed=kwargs.get("seed", os.environ["PL_GLOBAL_SEED"]))
        dataset['train'] = data_train['train']  
        dataset['val'] = data_train['test']
    elif "train" not in dataset.keys():
        dataset['train'] = []
        dataset['val'] = []

    debug_n = kwargs.get('debug_n', None)
    if debug_n is not None:
        dataset["train"] = dataset["train"].select(range(debug_n))
        dataset["val"] = dataset["val"].select(range(debug_n))
        dataset["test"] = dataset["test"].select(range(debug_n)) if "test" in dataset.keys() else None
    
    return dataset