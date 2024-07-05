from typing import Any, Dict, Optional, Tuple

import multiprocessing as mp
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# from torchvision.transforms import transforms
import os
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, Literal

import torch
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import pandas as pd
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase   
from tqdm import tqdm


T = TypeVar("T", bound=Union[Dataset, DatasetDict])

MODEL_BATCH_SIZE = 4
CHUNK_SIZE_GB = 2.0
MAX_SENTENCE_LEN = 256
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}



class GenerateActivationData(LightningDataModule):
    """
    LightningDataModule` for the activation data of a transformer model.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        print(self.hparams)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        self.probed_model = HookedTransformer.from_pretrained(self.hparams["model_name"]).to(self.hparams["model_device"])
        self.probed_model.name = self.hparams["model_name"]
        self.probed_model.eval()
        self.probed_model.requires_grad_(False)
        self.probed_model_conf = {
            'n_layers': self.probed_model.cfg.n_layers,
            'd_model': self.probed_model.cfg.d_model,
            'n_heads': self.probed_model.cfg.n_heads,
            'd_head': self.probed_model.cfg.d_head,
            'd_mlp': self.probed_model.cfg.d_mlp,
            'd_vocab': self.probed_model.cfg.d_vocab   
        }

    # def load_text_dataset(self, dataset_path: str, split: str = "train", streaming: bool = True):
    #         """
    #         Load a text dataset from Hugging Face's datasets library.
    #         """
    #         data = load_dataset(dataset_path, split=split, streaming=streaming)
    #         return

    def make_sentence_dataset(self, dataset_name: str, data_files: List, data_path: str = None, max_lines: int = 20_000, start_line: int = 0):
        """Returns a dataset from the Huggingface Datasets library."""
        # if dataset_name == "EleutherAI/pile":
        #     if not os.path.exists(os.path.join(data_path, "pile0.zst")):
        #         print("Downloading shard 0 of the Pile dataset (requires 50GB of disk space).")
        #         if not os.path.exists(os.path.join(data_path, "pile0.zst")):
        #             os.system(f"curl -o {data_path}/pile0.zst https://the-eye.eu/public/AI/pile/train/00.jsonl.zst ")
        #             os.system(f"unzstd {data_path}/pile0.zst")
        #     dataset = Dataset.from_list(list(self.read_from_pile("{data_path}/pile0", max_lines=max_lines, start_line=start_line)))
        # else:
        # os.environ["HF_DATASETS_CACHE"] = data_path
        # data_path = "/dlabdata1/masani/symbolic_probing/data"
        # data_files = {"train": ["train/00.jsonl.zst", "train/01.jsonl.zst"], "validation": "val.jsonl.zst", "test": "test.jsonl.zst"}
        self.dataset = load_dataset("monology/pile-uncopyrighted", data_files=data_files, cache_dir=data_path)
        return self.dataset

    def read_from_pile(self, address: str, max_lines: int = 100_000, start_line: int = 0):
        """Reads a file from the Pile dataset. Returns a generator."""
        with open(address, "r") as f:
            for i, line in enumerate(f):
                if i < start_line:
                    continue
                if i >= max_lines + start_line:
                    break
                yield json.loads(line)

    def save_activation_chunk(self, dataset, n_saved_chunks, dataset_folder):
        dataset_t = torch.cat(dataset, dim=0).to("cpu")
        os.makedirs(dataset_folder, exist_ok=True)
        with open(dataset_folder + "/" + str(n_saved_chunks) + ".pt", "wb") as f:
            torch.save(dataset_t, f)

    def make_activation_dataset(
        self,
        sentence_dataset: DataLoader = None,
        model: HookedTransformer = None,
        tensor_name: str = 'blocks.1.hook_mlp_out',
        activation_width: int = 512,
        dataset_folder: str = "activations",
        baukit: bool = False,
        chunk_size_gb: float = 2,
        layer: int = 2,
        n_chunks: int = 1,
        max_length: int = 1024,
        model_batch_size: int = 4,
        center_dataset: bool = False
    ) -> pd.DataFrame:
        if sentence_dataset is None:
            sentence_dataset = self.dataset
        if model is None:
            model = self.probed_model
        tokenizer = model.tokenizer
        device = next(model.parameters()).device
        dataset_folder = os.path.join(dataset_folder, model.name+"_"+tensor_name)
        # max_length = min(max_length, tokenizer.model_max_length, model.cfg.n_ctx) # model.pos_embed.W_pos.shape[0] perhaps?
        activities_per_input = 256
        n_saved_chunks = 0
        if os.path.exists(dataset_folder):
            # removing the folder and its contents and remaking it
            os.system(f"rm -r {dataset_folder}")
        os.makedirs(os.path.join(dataset_folder), exist_ok=True)
        
        generator = torch.Generator(device=device).manual_seed(42)

        
        data = {'activations': []} # Store activations here, add logits if you want to save them as well
        data_size = 0 # Keep track of the data size for periodic saving

        with torch.no_grad():
            for split, split_dataset in sentence_dataset.items():
                split_dataloader = DataLoader(split_dataset, batch_size=model_batch_size, shuffle=False)
                for batch_idx, batch in tqdm(enumerate(split_dataloader)):
                    tokenized_batch = model.to_tokens(batch['text'])
                    # tokenized_batch = tokenizer(batch['text'], padding=True, max_length=max_length, truncation=True, return_tensors="pt")
                    (logits, loss), cache = model.run_with_cache(tokenized_batch, return_type='both')
                    # model.tokenizer.decode(logits.argmax(dim=-1)[0]) # to decode the logits and see how the predictions look like
                    
                    labels = tokenized_batch[:, 1:]
                    # logits = logits[:, :-1, :]
                    activations = cache[tensor_name][:, :-1, :]
                    # subsample 256 activations from each of the 1024 context window without replacement
                    # compute NLL loss for each of the 256 activations
                    labels_not_pad_mask = (labels != tokenizer.pad_token_id)
                    labels = labels[labels_not_pad_mask]
                    # logits = logits[labels_not_pad_mask]
                    activations = activations[labels_not_pad_mask]
                    perm = torch.randperm(labels.shape[0], generator=generator, device=device)
                    
                    activations_subsampled = activations[perm[:activities_per_input * model_batch_size]].detach().to(torch.float16).reshape(-1, activations.shape[-1]).cpu()
                    data_size += activations_subsampled.nelement() * activations_subsampled.element_size()
                    # data_size += activations_subsampled.nbytes #for numpy
                    # logits_subsampled = logits[perm[:activities_per_input * model_batch_size]].detach().to(torch.float16).reshape(-1, logits.shape[-1]).cpu().numpy()

                    # activation_data.append(activations_subsampled)
                    # logit_data.append(logits_subsampled)
                    # data_size += activations_subsampled.nelement() * activations_subsampled.element_size() + logits_subsampled.nelement() * logits_subsampled.element_size()
                    # data = {'activations': activations_subsampled, 'logits': logits_subsampled} # use this if you want to save the logits as well
                    # data = {'activations': activations_subsampled} # use this if you want to save only the activations
                    data['activations'].extend(activations_subsampled)

                    # for key in data:
                    #     data[key].nelement() * data[key].element_size()
                    # for key in data:
                        # data_size += data[key].nbytes()
                        # data_size += data[key].nelement() * data[key].element_size()


                    if data_size >= 2 * (1024 ** 3):  # Assuming 2GB before saving and resetting 2 * (1024 ** 3)
                        self.save_dataset_chunk(data, dataset_folder, split)
                        for key in data:
                            data[key] = []
                        data_size = 0
                    
                    # # for debugging
                    # if batch_idx > 10:
                    #     break

                # Save any remaining data that didn't reach the threshold
                if data_size > 0:
                    self.save_dataset_chunk(data, dataset_folder, split)

            # Concatenate all chunks and make one final dataset
            final_datasets = self.concatenate_datasets(sentence_dataset, dataset_folder, tokenizer, max_length)
            return final_datasets

    # def tensor_to_list(self, tensor):
    #     """Converts a tensor to a list of values, ensuring compatibility with the datasets library."""
    #     return tensor.detach().cpu().tolist()

    def save_dataset_chunk(self, data, dataset_folder, split_name):
        """Saves a chunk of the dataset to disk using the datasets library, tailored for PyTorch tensors."""
        # Prepare the data by converting tensors to lists of values
        # activations_lists = [self.tensor_to_list(a) for a in activation_data]
        # logits_lists = [self.tensor_to_list(l) for l in logit_data]
        
        # for key in data:
        #     data[key] = self.tensor_to_list(data[key])

        # Prepare a list of dictionaries (each dictionary corresponds to a single example in the dataset)
        # data = [{'activations': act, 'logits': log} for act, log in zip(activations_lists, logits_lists)]

        # Create a Hugging Face dataset from the list of dictionaries
        dataset_chunk = Dataset.from_dict(data)
        
        if not os.path.exists(os.path.join(dataset_folder, f"{split_name}")):
            os.makedirs(os.path.join(dataset_folder, f"{split_name}"))
        last_chunk_id = len(os.listdir(os.path.join(dataset_folder, f"{split_name}")))
        save_path = os.path.join(dataset_folder, f"{split_name}", f"chunk_{last_chunk_id}")

        dataset_chunk.save_to_disk(save_path)
        print(f"chunk {last_chunk_id} saved to {save_path}")


        # if os.path.exists(dataset_path):
        #     dataset_dict = DatasetDict.load_from_disk(dataset_path)
        #     if split_name in dataset_dict:
        #         # Concatenate the new chunk to the existing dataset split
        #         # dataset_dict[split_name] = DatasetDict({split_name: dataset_dict[split_name].concatenate(dataset_chunk)})
        #         dataset_dict[split_name] = concatenate_datasets([dataset_dict[split_name], dataset_chunk])
        #     else:
        #         # Add new split with the chunk
        #         dataset_dict[split_name] = dataset_chunk
        # else:
            # # Initialize a new dataset with the chunk
            # dataset_dict = DatasetDict({split_name: dataset_chunk})

        # Save the dataset
        # save_path = os.path.join(dataset_path, f"{split}_chunk_{num_chunks}")
        # dataset_dict.save_to_disk(dataset_path)
        # print(f"chunk {num_chunks} saved to {save_path}")
        # print(f"Updated dataset for {split_name} split at {dataset_path}")

    def concatenate_datasets(self, sentence_dataset, dataset_folder, tokenizer, max_length):
        # Load and concatenate all chunks for each split
        # making a final dataset with all the chunks and splits
        final_dataset = DatasetDict()
        for split in sentence_dataset.keys():
            data_split_path = os.path.join(dataset_folder, split)
            num_chunks = len(os.listdir(data_split_path))
            for chunk_num in range(num_chunks):
                chunk_path = os.path.join(data_split_path, f"chunk_{chunk_num}")
                if chunk_num == 0:
                    loaded_dset = Dataset.load_from_disk(chunk_path)
                else:
                    loaded_dset = concatenate_datasets([loaded_dset, Dataset.load_from_disk(chunk_path)])
            final_dataset[split] = loaded_dset
        
        # Optionally, save the concatenated final datasets to disk
        final_path = os.path.join(dataset_folder, "final")
        final_dataset.save_to_disk(final_path)

        return final_datasets


            # tokenized_dataset = self.chunk_and_tokenize(split_dataset, tokenizer, max_length=max_length)
            # output = model.run_with_cache(tokenized_dataset["input_ids"].to(device), stop_at_layer=layer + 1)
            # for sentence_idx, sentence in tqdm(enumerate(sentence_dataset)):
            #     tokenized_sentence = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
            #     batch = tokenized_sentence["input_ids"].to(device)

            #     _, cache = model.run_with_cache(batch, stop_at_layer=layer + 1)
            #     mlp_activation_data = (
            #         cache[tensor_name].to(device).to(torch.float16)
            #     )  # NOTE: could do all layers at once, but currently just doing 1 layer
            #     mlp_activation_data = rearrange(mlp_activation_data, "b s n -> (b s) n")

            #     dataset.append(mlp_activation_data)
            #     if len(dataset) >= actives_per_chunk:
            #         if center_dataset:
            #             if n_saved_chunks == 0:
            #                 chunk_mean = torch.mean(torch.cat(dataset), dim=0)
            #             dataset = [x - chunk_mean for x in dataset]
                        
            #         # Need to save, restart the list
            #         save_activation_chunk(dataset, n_saved_chunks, dataset_folder)
            #         n_saved_chunks += 1
            #         print(f"Saved chunk {n_saved_chunks} of activations, total size:  {batch_idx * activation_size} ")
            #         dataset = []
            #         if n_saved_chunks == n_chunks:
            #             break

            # if n_saved_chunks < n_chunks:
            #     save_activation_chunk(dataset, n_saved_chunks, dataset_folder)
            #     print(f"Saved undersized chunk {n_saved_chunks} of activations, total size:  {batch_idx * activation_size} ")
    
    # # Nora's Code from https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/data.py
    # def chunk_and_tokenize(self,
    #     data: T,
    #     tokenizer: PreTrainedTokenizerBase,
    #     *,
    #     format: str = "torch",
    #     num_proc: int = min(mp.cpu_count() // 2, 8),
    #     text_key: str = "text",
    #     max_length: int = 2048,
    #     return_final_batch: bool = False,
    #     load_from_cache_file: bool = True,
    # )-> Tuple[T, float]:
    #     """Perform GPT-style chunking and tokenization on a dataset.

    #     The resulting dataset will consist entirely of chunks exactly `max_length` tokens
    #     long. Long sequences will be split into multiple chunks, and short sequences will
    #     be merged with their neighbors, using `eos_token` as a separator. The fist token
    #     will also always be an `eos_token`.

    #     Args:
    #         data: The dataset to chunk and tokenize.
    #         tokenizer: The tokenizer to use.
    #         format: The format to return the dataset in, passed to `Dataset.with_format`.
    #         num_proc: The number of processes to use for tokenization.
    #         text_key: The key in the dataset to use as the text to tokenize.
    #         max_length: The maximum length of a batch of input ids.
    #         return_final_batch: Whether to return the final batch, which may be smaller
    #             than the others.
    #         load_from_cache_file: Whether to load from the cache file.

    #     Returns:
    #         * The chunked and tokenized dataset.
    #         * The ratio of nats to bits per byte see https://arxiv.org/pdf/2101.00027.pdf,
    #             section 3.1.
    #     """

    #     def _tokenize_fn(x: Dict[str, list]):
    #         chunk_size = min(tokenizer.model_max_length, max_length)  # tokenizer max length is 1024 for gpt2
    #         sep = tokenizer.eos_token or "<|endoftext|>"
    #         joined_text = sep.join([""] + x[text_key])
    #         output = tokenizer(
    #             # Concatenate all the samples together, separated by the EOS token.
    #             joined_text,  # start with an eos token
    #             max_length=chunk_size,
    #             return_attention_mask=False,
    #             return_overflowing_tokens=True,
    #             truncation=True,
    #         )

    #         if overflow := output.pop("overflowing_tokens", None):
    #             # Slow Tokenizers return unnested lists of ints
    #             assert isinstance(output["input_ids"][0], int)

    #             # Chunk the overflow into batches of size `chunk_size`
    #             chunks = [output["input_ids"]] + [
    #                 overflow[i * chunk_size : (i + 1) * chunk_size] for i in range(math.ceil(len(overflow) / chunk_size))
    #             ]
    #             output = {"input_ids": chunks}

    #         total_tokens = sum(len(ids) for ids in output["input_ids"])
    #         total_bytes = len(joined_text.encode("utf-8"))

    #         if not return_final_batch:
    #             # We know that the last sample will almost always be less than the max
    #             # number of tokens, and we don't want to pad, so we just drop it.
    #             output = {k: v[:-1] for k, v in output.items()}

    #         output_batch_size = len(output["input_ids"])

    #         if output_batch_size == 0:
    #             raise ValueError(
    #                 "Not enough data to create a single batch complete batch."
    #                 " Either allow the final batch to be returned,"
    #                 " or supply more data."
    #             )

    #         # We need to output this in order to compute the number of bits per byte
    #         div, rem = divmod(total_tokens, output_batch_size)
    #         output["length"] = [div] * output_batch_size
    #         output["length"][-1] += rem

    #         div, rem = divmod(total_bytes, output_batch_size)
    #         output["bytes"] = [div] * output_batch_size
    #         output["bytes"][-1] += rem

    #         return output

    #     def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> List[str]:
    #         """Get a single list of columns in a `Dataset` or `DatasetDict`.

    #         We assert the columms are the same across splits if it's a `DatasetDict`.

    #         Args:
    #             dataset: The dataset to get the columns from.

    #         Returns:
    #             A list of columns.
    #         """
    #         if isinstance(dataset, DatasetDict):
    #             cols_by_split = dataset.column_names.values()
    #             columns = next(iter(cols_by_split))
    #             if not all(cols == columns for cols in cols_by_split):
    #                 raise ValueError("All splits must have the same columns")

    #             return columns

    #         return dataset.column_names

    #     # take 2048 texts from Pile, concat them together by appending eos token, tokenize them, 
    #     # chunck them into chunk_size=256 tokens, and return the tokenized texts as a much larger than 2048 batch of 
    #     # equal length tokenized texts

    #     data = data.map(
    #         _tokenize_fn,
    #         # Batching is important for ensuring that we don't waste tokens
    #         # since we always throw away the last element of the batch we
    #         # want to keep the batch size as large as possible
    #         batched=True,
    #         batch_size=2048,
    #         num_proc=num_proc,
    #         remove_columns=get_columns_all_equal(data),
    #         load_from_cache_file=load_from_cache_file,
    #     )
    #     total_bytes: float = sum(data["bytes"])
    #     total_tokens: float = sum(data["length"])
    #     return data.with_format(format, columns=["input_ids"]), (total_tokens / total_bytes) / math.log(2)


        

    # # End Nora's Code from https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/data.py



    
if __name__ == "__main__":

    generate_activation_data_hparams= {
    'model_name': "gelu-2l",
    'model_device': 'cuda:1',
    }
    
    generate_activation_data_class = GenerateActivationData(**generate_activation_data_hparams)
    
    # loading text dataset
    data_path = os.path.join(os.getcwd(), "symbolic_probing", "data")
    data_files = {"train": ["train/00.jsonl.zst", "train/01.jsonl.zst"], "validation": "val.jsonl.zst", "test": "test.jsonl.zst"}
    dataset = generate_activation_data_class.make_sentence_dataset("monology/pile-uncopyrighted", data_files=data_files, data_path=data_path, max_lines=20_000, start_line=0)
    print(dataset)

    # make activation dataset
    dataset_folder = os.path.join(os.getcwd(), "symbolic_probing", "data", "activation_data")
    generate_activation_data_class.make_activation_dataset(dataset, dataset_folder=dataset_folder)

    




