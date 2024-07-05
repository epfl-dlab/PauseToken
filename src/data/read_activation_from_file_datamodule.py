from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from datasets import Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm
import os
import code
from omegaconf import DictConfig


class ReadActivationFromFileDataModule(LightningDataModule):
    """`LightningDataModule` for the PVR dataset.
    """

    def __init__(self, **kwargs) -> None:
        self,
        # data_dir: str = "data/",
        # train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        # batch_size: int = 64,
        # num_workers: int = 0,
        # pin_memory: bool = False,

        super().__init__()       
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = kwargs['batch_size']

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices. This is necessary for multi-GPU training.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            generator = torch.Generator().manual_seed(self.hparams.seed)
            dataset = self.__load_dataset(self.hparams.path)
            if len(dataset.keys()) < 3: 
                dataset = dataset[list(dataset.keys())[0]]
                self.data_train, self.data_val, self.data_test = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_val_test_split,
                    generator=generator,
                )
            else:
                self.data_train = dataset['train']
                self.data_val = dataset['validation']
                self.data_test = dataset['test']
    

    def __load_dataset(self, data_dir: str) -> Dataset:
        """Load the dataset from the data directory.

        :param data_dir: The directory containing the dataset.
        :return: The dataset.
        """
        # Load the dataset from the data directory
        if self.hparams.format=='chunks':
            dataset = self.__load_dataset_from_chunks(data_dir)
        elif self.hparams.format=='one_file':
            dataset = self.__load_dataset_from_one_file(data_dir)
        else:
            raise ValueError(f"Invalid format: {self.hparams.format}")
        return dataset


    def __load_dataset_from_chunks(self, data_dir: str) -> Dataset:
        """Load the dataset parts from the data directory.
        merge them into a file.
        """
        # Load and concatenate all chunks for each split
        # making a final dataset with all the chunks and splits
        # removing the final directory if it exists
        final_path = os.path.join(data_dir, "final")
        if os.path.exists(final_path):
            os.system(f"rm -r {final_path}")
        final_dataset = DatasetDict()
        num_chunks = len(os.listdir(data_dir))
        for split in os.listdir(data_dir):
            data_split_path = os.path.join(data_dir, split)
            num_chunks = len(os.listdir(data_split_path))
            # for chunk_num in range(num_chunks):
            #     chunk_path = os.path.join(data_split_path, f"chunk_{chunk_num}")
            dir_list = os.listdir(data_split_path)
            dir_list.sort()
            chunk_num = 0
            for chunk_name in tqdm(dir_list, desc=f"Loading {split}"):
                chunk_path = os.path.join(data_split_path, chunk_name)
                if chunk_num == 0:
                    loaded_dset = Dataset.load_from_disk(chunk_path)
                    chunk_num += 1
                else:
                    if os.path.exists(chunk_path+"dataset_info.json"):
                        loaded_dset = concatenate_datasets([loaded_dset, Dataset.load_from_disk(chunk_path)])
                        chunk_num += 1
                    else:
                        print(f"Skipping {chunk_path}, file broken")
                if chunk_num == self.hparams.max_num_chunks:
                    break
            final_dataset[split] = loaded_dset
        
        # Optionally, save the concatenated final datasets to disk
        final_dataset.save_to_disk(final_path)

        return final_dataset
        
    def __load_dataset_from_one_file(self, data_dir: str) -> Dataset:
        """Load the dataset from a single file.
        """
        try:
            dataset = Dataset.load_from_disk(data_dir + "/final")
        except:
            dataset = DatasetDict.load_from_disk(data_dir + "/final")
        return dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        # TODO: Implement
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # TODO: Implement
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass



        

@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="read_activation_from_file.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    data_module = ReadActivationFromFileDataModule(**cfg)
    data_module.setup()
    dl = data_module.train_dataloader()
    for x in dl:
        print(x)
        break

if __name__ == "__main__":
    main()
