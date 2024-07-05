# from typing import Any, Dict, Optional, Tuple

# import multiprocessing
# import torch
# from lightning import LightningDataModule
# from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# from torchvision.transforms import transforms
# import os
# from typing import Any, Iterator, cast

# import torch
# from datasets import load_dataset
# from transformer_lens import HookedTransformer

# class TransformerActivationDataModule(LightningDataModule):
#     """
#     LightningDataModule` for the activation data of a transformer model.
#     """

#     def __init__(
#         self,
#         data_dir: str = "data/",
#         batch_size: int = 64,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#     ) -> None:
#         super().__init__()

#         # this line allows to access init params with 'self.hparams' attribute
#         # also ensures init params will be stored in ckpt
#         self.save_hyperparameters(logger=False)

#         self.data_train: Optional[Dataset] = None
#         self.data_val: Optional[Dataset] = None
#         self.data_test: Optional[Dataset] = None

#         self.batch_size_per_device = batch_size

#         self.probed_model = HookedTransformer.from_pretrained(self.hparams["model_name"]).to(DTYPES[self.hparams["enc_dtype"]]).to(self.hparams["device"])
#         self.probed_model.eval()
#         self.probed_model.requires_grad_(False)
#         self.probed_model_conf = {
#             'n_layers': self.probed_model.cfg.n_layers,
#             'd_model': self.probed_model.cfg.d_model,
#             'n_heads': self.probed_model.cfg.n_heads,
#             'd_head': self.probed_model.cfg.d_head,
#             'd_mlp': self.probed_model.cfg.d_mlp,
#             'd_vocab': self.probed_model.cfg.d_vocab   
#         }

#     def prepare_data(self) -> None:
#         """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
#         within a single process on CPU, so you can safely add your downloading logic within. In
#         case of multi-node training, the execution of this hook depends upon
#         `self.prepare_data_per_node()`.

#         Do not use it to assign state (self.x = y).
#         """


#     def setup(self, stage: Optional[str] = None) -> None:
#         """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

#         This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
#         `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
#         `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
#         `self.setup()` once the data is prepared and available for use.

#         :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
#         """
#         # Divide batch size by the number of devices.
#         if self.trainer is not None:
#             if self.hparams.batch_size % self.trainer.world_size != 0:
#                 raise RuntimeError(
#                     f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
#                 )
#             self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

#         # load and split datasets only if not loaded already
#         if not self.data_train and not self.data_val and not self.data_test:
#             trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
#             testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
#             dataset = ConcatDataset(datasets=[trainset, testset])
#             self.data_train, self.data_val, self.data_test = random_split(
#                 dataset=dataset,
#                 lengths=self.hparams.train_val_test_split,
#                 generator=torch.Generator().manual_seed(42),
#             )

#     def train_dataloader(self) -> DataLoader[Any]:
#         """Create and return the train dataloader.

#         :return: The train dataloader.
#         """
#         return DataLoader(
#             dataset=self.data_train,
#             batch_size=self.batch_size_per_device,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=True,
#         )

#     def val_dataloader(self) -> DataLoader[Any]:
#         """Create and return the validation dataloader.

#         :return: The validation dataloader.
#         """
#         return DataLoader(
#             dataset=self.data_val,
#             batch_size=self.batch_size_per_device,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=False,
#         )

#     def test_dataloader(self) -> DataLoader[Any]:
#         """Create and return the test dataloader.

#         :return: The test dataloader.
#         """
#         return DataLoader(
#             dataset=self.data_test,
#             batch_size=self.batch_size_per_device,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=False,
#         )

#     def load_text_dataset(self, dataset_path: str, split: str = "train", streaming: bool = True):
#         """
#         Load a text dataset from Hugging Face's datasets library.
#         """
#         data = load_dataset(dataset_path, split=split, streaming=streaming)
#         return



#     def read_from_pile(address: str, max_lines: int = 100_000, start_line: int = 0):
#         """Reads a file from the Pile dataset. Returns a generator."""
#         with open(address, "r") as f:
#             for i, line in enumerate(f):
#                 if i < start_line:
#                     continue
#                 if i >= max_lines + start_line:
#                     break
#                 yield json.loads(line)


#     def make_sentence_dataset(dataset_name: str, max_lines: int = 20_000, start_line: int = 0):
#         """Returns a dataset from the Huggingface Datasets library."""
#         if dataset_name == "EleutherAI/pile":
#             if not os.path.exists("pile0"):
#                 print("Downloading shard 0 of the Pile dataset (requires 50GB of disk space).")
#                 if not os.path.exists("pile0.zst"):
#                     os.system("curl https://the-eye.eu/public/AI/pile/train/00.jsonl.zst > pile0.zst")
#                     os.system("unzstd pile0.zst")
#             dataset = Dataset.from_list(list(read_from_pile("pile0", max_lines=max_lines, start_line=start_line)))
#         else:
#             dataset = load_dataset(dataset_name, split="train")#, split=f"train[{start_line}:{start_line + max_lines}]")
#         return dataset


#     # Nora's Code from https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/data.py
#     def chunk_and_tokenize(
#         data: T,
#         tokenizer: PreTrainedTokenizerBase,
#         *,
#         format: str = "torch",
#         num_proc: int = min(mp.cpu_count() // 2, 8),
#         text_key: str = "text",
#         max_length: int = 2048,
#         return_final_batch: bool = False,
#         load_from_cache_file: bool = True,
#     ) -> Tuple[T, float]:
#         """Perform GPT-style chunking and tokenization on a dataset.

#         The resulting dataset will consist entirely of chunks exactly `max_length` tokens
#         long. Long sequences will be split into multiple chunks, and short sequences will
#         be merged with their neighbors, using `eos_token` as a separator. The fist token
#         will also always be an `eos_token`.

#         Args:
#             data: The dataset to chunk and tokenize.
#             tokenizer: The tokenizer to use.
#             format: The format to return the dataset in, passed to `Dataset.with_format`.
#             num_proc: The number of processes to use for tokenization.
#             text_key: The key in the dataset to use as the text to tokenize.
#             max_length: The maximum length of a batch of input ids.
#             return_final_batch: Whether to return the final batch, which may be smaller
#                 than the others.
#             load_from_cache_file: Whether to load from the cache file.

#         Returns:
#             * The chunked and tokenized dataset.
#             * The ratio of nats to bits per byte see https://arxiv.org/pdf/2101.00027.pdf,
#                 section 3.1.
#         """

#         def _tokenize_fn(x: Dict[str, list]):
#             chunk_size = min(tokenizer.model_max_length, max_length)  # tokenizer max length is 1024 for gpt2
#             sep = tokenizer.eos_token or "<|endoftext|>"
#             joined_text = sep.join([""] + x[text_key])
#             output = tokenizer(
#                 # Concatenate all the samples together, separated by the EOS token.
#                 joined_text,  # start with an eos token
#                 max_length=chunk_size,
#                 return_attention_mask=False,
#                 return_overflowing_tokens=True,
#                 truncation=True,
#             )

#             if overflow := output.pop("overflowing_tokens", None):
#                 # Slow Tokenizers return unnested lists of ints
#                 assert isinstance(output["input_ids"][0], int)

#                 # Chunk the overflow into batches of size `chunk_size`
#                 chunks = [output["input_ids"]] + [
#                     overflow[i * chunk_size : (i + 1) * chunk_size] for i in range(math.ceil(len(overflow) / chunk_size))
#                 ]
#                 output = {"input_ids": chunks}

#             total_tokens = sum(len(ids) for ids in output["input_ids"])
#             total_bytes = len(joined_text.encode("utf-8"))

#             if not return_final_batch:
#                 # We know that the last sample will almost always be less than the max
#                 # number of tokens, and we don't want to pad, so we just drop it.
#                 output = {k: v[:-1] for k, v in output.items()}

#             output_batch_size = len(output["input_ids"])

#             if output_batch_size == 0:
#                 raise ValueError(
#                     "Not enough data to create a single batch complete batch."
#                     " Either allow the final batch to be returned,"
#                     " or supply more data."
#                 )

#             # We need to output this in order to compute the number of bits per byte
#             div, rem = divmod(total_tokens, output_batch_size)
#             output["length"] = [div] * output_batch_size
#             output["length"][-1] += rem

#             div, rem = divmod(total_bytes, output_batch_size)
#             output["bytes"] = [div] * output_batch_size
#             output["bytes"][-1] += rem

#             return output

#         data = data.map(
#             _tokenize_fn,
#             # Batching is important for ensuring that we don't waste tokens
#             # since we always throw away the last element of the batch we
#             # want to keep the batch size as large as possible
#             batched=True,
#             batch_size=2048,
#             num_proc=num_proc,
#             remove_columns=get_columns_all_equal(data),
#             load_from_cache_file=load_from_cache_file,
#         )
#         total_bytes: float = sum(data["bytes"])
#         total_tokens: float = sum(data["length"])
#         return data.with_format(format, columns=["input_ids"]), (total_tokens / total_bytes) / math.log(2)


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


#     # End Nora's Code from https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/data.py

#     def make_activation_dataset(
#         sentence_dataset: DataLoader,
#         model: HookedTransformer,
#         tensor_name: str,
#         activation_width: int,
#         dataset_folder: str,
#         baukit: bool = False,
#         chunk_size_gb: float = 2,
#         device: torch.device = torch.device("cuda:0"),
#         layer: int = 2,
#         n_chunks: int = 1,
#         max_length: int = 256,
#         model_batch_size: int = 4,
#         center_dataset: bool = False
#     ) -> pd.DataFrame:
#         print(f"Running model and saving activations to {dataset_folder}")
#         with torch.no_grad():
#             chunk_size = chunk_size_gb * (2**30)  # 2GB
#             activation_size = (
#                 activation_width * 2 * model_batch_size * max_length
#             )  # 3072 mlp activations, 2 bytes per half, 1024 context window
#             actives_per_chunk = chunk_size // activation_size
#             dataset = []
#             n_saved_chunks = 0
#             for batch_idx, batch in tqdm(enumerate(sentence_dataset)):
#                 batch = batch["input_ids"].to(device)
#                 if baukit:
#                     # Don't have nanoGPT models integrated with transformer_lens so using baukit for activations
#                     with Trace(model, tensor_name) as ret:
#                         _ = model(batch)
#                         mlp_activation_data = ret.output
#                         mlp_activation_data = rearrange(mlp_activation_data, "b s n -> (b s) n").to(torch.float16).to(device)
#                         mlp_activation_data = nn.functional.gelu(mlp_activation_data)
#                 else:
#                     _, cache = model.run_with_cache(batch, stop_at_layer=layer + 1)
#                     mlp_activation_data = (
#                         cache[tensor_name].to(device).to(torch.float16)
#                     )  # NOTE: could do all layers at once, but currently just doing 1 layer
#                     mlp_activation_data = rearrange(mlp_activation_data, "b s n -> (b s) n")

#                 dataset.append(mlp_activation_data)
#                 if len(dataset) >= actives_per_chunk:
#                     if center_dataset:
#                         if n_saved_chunks == 0:
#                             chunk_mean = torch.mean(torch.cat(dataset), dim=0)
#                         dataset = [x - chunk_mean for x in dataset]
                        
#                     # Need to save, restart the list
#                     save_activation_chunk(dataset, n_saved_chunks, dataset_folder)
#                     n_saved_chunks += 1
#                     print(f"Saved chunk {n_saved_chunks} of activations, total size:  {batch_idx * activation_size} ")
#                     dataset = []
#                     if n_saved_chunks == n_chunks:
#                         break

#             if n_saved_chunks < n_chunks:
#                 save_activation_chunk(dataset, n_saved_chunks, dataset_folder)
#                 print(f"Saved undersized chunk {n_saved_chunks} of activations, total size:  {batch_idx * activation_size} ")




# # import os
# # from typing import Any, Iterator, cast

# # import torch
# # from datasets import load_dataset
# # from torch.utils.data import DataLoader
# # from transformer_lens import HookedTransformer


# # class ActivationData:
# #     """
# #     Class for streaming tokens and generating and storing activations
# #     while training SAEs.
# #     cfg: config object with the following attributes:
# #         - dataset_path: path to the dataset
# #         - use_cached_activations: whether to use cached activations
# #         - cached_activations_path: path to the directory containing cached activations
# #         - total_training_tokens: total number of tokens to train on
# #         - n_batches_in_buffer: number of batches to store in the buffer
# #         - store_batch_size: number of tokens to store in the buffer at a time
# #         - train_batch_size: number of tokens to train on at a time
# #         - context_size: number of tokens in the context
# #         - d_in: input dimensionality
# #         - hook_point_layer: layer to hook into
# #         - hook_point_head_index: head index to hook into
# #         - hook_point: name of the hook
# #         - device: device to store the activations on
# #         - dtype: data type to store the activations in
# #     model: the model to generate activations from
# #     create_dataloader: whether to create a dataloader
# #     """

# #     def __init__(
# #         self,
# #         cfg: Any,
# #         model: HookedTransformer,
# #         create_dataloader: bool = True,
# #     ):
# #         self.cfg = cfg
# #         self.model = model
# #         self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=True)
# #         self.iterable_dataset = iter(self.dataset)

# #         # Check if dataset is tokenized
# #         dataset_sample = next(self.iterable_dataset)
# #         self.cfg.is_dataset_tokenized = "tokens" in dataset_sample.keys()
# #         print(
# #             f"Dataset is {'tokenized' if self.cfg.is_dataset_tokenized else 'not tokenized'}! Updating config."
# #         )
# #         self.iterable_dataset = iter(self.dataset)  # Reset iterator after checking

# #         if self.cfg.use_cached_activations:  # EDIT: load from multi-layer acts
# #             assert self.cfg.cached_activations_path is not None  # keep pyright happy
# #             # Sanity check: does the cache directory exist?
# #             assert os.path.exists(
# #                 self.cfg.cached_activations_path
# #             ), f"Cache directory {self.cfg.cached_activations_path} does not exist. Consider double-checking your dataset, model, and hook names."

# #             self.next_cache_idx = 0  # which file to open next
# #             self.next_idx_within_buffer = 0  # where to start reading from in that file

# #             # Check that we have enough data on disk
# #             first_buffer = torch.load(f"{self.cfg.cached_activations_path}/0.pt")
# #             buffer_size_on_disk = first_buffer.shape[0]
# #             n_buffers_on_disk = len(os.listdir(self.cfg.cached_activations_path))
# #             # Note: we're assuming all files have the same number of tokens
# #             # (which seems reasonable imo since that's what our script does)
# #             n_activations_on_disk = buffer_size_on_disk * n_buffers_on_disk
# #             assert (
# #                 n_activations_on_disk > self.cfg.total_training_tokens
# #             ), f"Only {n_activations_on_disk/1e6:.1f}M activations on disk, but cfg.total_training_tokens is {self.cfg.total_training_tokens/1e6:.1f}M."

# #             # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)

# #         if create_dataloader:
# #             # fill buffer half a buffer, so we can mix it with a new buffer
# #             self.storage_buffer = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
# #             self.dataloader = self.get_data_loader()

# #     def get_batch_tokens(self):
# #         """
# #         Streams a batch of tokens from a dataset.
# #         """

# #         batch_size = self.cfg.store_batch_size
# #         context_size = self.cfg.context_size
# #         device = self.cfg.device

# #         batch_tokens = torch.zeros(
# #             size=(0, context_size), device=device, dtype=torch.long, requires_grad=False
# #         )

# #         current_batch = []
# #         current_length = 0

# #         # pbar = tqdm(total=batch_size, desc="Filling batches")
# #         while batch_tokens.shape[0] < batch_size:
# #             if not self.cfg.is_dataset_tokenized:
# #                 s = next(self.iterable_dataset)["text"]
# #                 tokens = self.model.to_tokens(
# #                     s,
# #                     truncate=True,
# #                     move_to_device=True,
# #                 ).squeeze(0)
# #                 assert (
# #                     len(tokens.shape) == 1
# #                 ), f"tokens.shape should be 1D but was {tokens.shape}"
# #             else:
# #                 tokens = torch.tensor(
# #                     next(self.iterable_dataset)["tokens"],
# #                     dtype=torch.long,
# #                     device=device,
# #                     requires_grad=False,
# #                 )
# #             token_len = tokens.shape[0]

# #             # TODO: Fix this so that we are limiting how many tokens we get from the same context.
# #             assert self.model.tokenizer is not None  # keep pyright happy
# #             bos_token_id_tensor = torch.tensor(
# #                 [self.model.tokenizer.bos_token_id],
# #                 device=tokens.device,
# #                 dtype=torch.long,
# #             )
# #             while token_len > 0 and batch_tokens.shape[0] < batch_size:
# #                 # Space left in the current batch
# #                 space_left = context_size - current_length

# #                 # If the current tokens fit entirely into the remaining space
# #                 if token_len <= space_left:
# #                     current_batch.append(tokens[:token_len])
# #                     current_length += token_len
# #                     break

# #                 else:
# #                     # Take as much as will fit
# #                     current_batch.append(tokens[:space_left])

# #                     # Remove used part, add BOS
# #                     tokens = tokens[space_left:]
# #                     tokens = torch.cat(
# #                         (
# #                             bos_token_id_tensor,
# #                             tokens,
# #                         ),
# #                         dim=0,
# #                     )

# #                     token_len -= space_left
# #                     token_len += 1
# #                     current_length = context_size

# #                 # If a batch is full, concatenate and move to next batch
# #                 if current_length == context_size:
# #                     full_batch = torch.cat(current_batch, dim=0)
# #                     batch_tokens = torch.cat(
# #                         (batch_tokens, full_batch.unsqueeze(0)), dim=0
# #                     )
# #                     current_batch = []
# #                     current_length = 0

# #             # pbar.n = batch_tokens.shape[0]
# #             # pbar.refresh()
# #         return batch_tokens[:batch_size]

# #     def get_activations(self, batch_tokens: torch.Tensor, get_loss: bool = False):
# #         """
# #         Returns activations of shape (batches, context, num_layers, d_in)
# #         """
# #         layers = (
# #             self.cfg.hook_point_layer
# #             if isinstance(self.cfg.hook_point_layer, list)
# #             else [self.cfg.hook_point_layer]
# #         )
# #         act_names = [self.cfg.hook_point.format(layer=layer) for layer in layers]
# #         hook_point_max_layer = max(layers)
# #         if self.cfg.hook_point_head_index is not None:
# #             layerwise_activations = self.model.run_with_cache(
# #                 batch_tokens,
# #                 names_filter=act_names,
# #                 stop_at_layer=hook_point_max_layer + 1,
# #             )[1]
# #             activations_list = [
# #                 layerwise_activations[act_name][:, :, self.cfg.hook_point_head_index]
# #                 for act_name in act_names
# #             ]
# #         else:
# #             layerwise_activations = self.model.run_with_cache(
# #                 batch_tokens,
# #                 names_filter=act_names,
# #                 stop_at_layer=hook_point_max_layer + 1,
# #             )[1]
# #             activations_list = [
# #                 layerwise_activations[act_name] for act_name in act_names
# #             ]

# #         # Stack along a new dimension to keep separate layers distinct
# #         stacked_activations = torch.stack(activations_list, dim=2)

# #         return stacked_activations

# #     def get_buffer(self, n_batches_in_buffer: int):
# #         context_size = self.cfg.context_size
# #         batch_size = self.cfg.store_batch_size
# #         d_in = self.cfg.d_in
# #         total_size = batch_size * n_batches_in_buffer
# #         num_layers = (
# #             len(self.cfg.hook_point_layer)
# #             if isinstance(self.cfg.hook_point_layer, list)
# #             else 1
# #         )  # Number of hook points or layers

# #         if self.cfg.use_cached_activations:
# #             # Load the activations from disk
# #             buffer_size = total_size * context_size
# #             # Initialize an empty tensor with an additional dimension for layers
# #             new_buffer = torch.zeros(
# #                 (buffer_size, num_layers, d_in),
# #                 dtype=self.cfg.dtype,
# #                 device=self.cfg.device,
# #             )
# #             n_tokens_filled = 0

# #             # Assume activations for different layers are stored separately and need to be combined
# #             while n_tokens_filled < buffer_size:
# #                 if not os.path.exists(
# #                     f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt"
# #                 ):
# #                     print(
# #                         "\n\nWarning: Ran out of cached activation files earlier than expected."
# #                     )
# #                     print(
# #                         f"Expected to have {buffer_size} activations, but only found {n_tokens_filled}."
# #                     )
# #                     if buffer_size % self.cfg.total_training_tokens != 0:
# #                         print(
# #                             "This might just be a rounding error — your batch_size * n_batches_in_buffer * context_size is not divisible by your total_training_tokens"
# #                         )
# #                     print(f"Returning a buffer of size {n_tokens_filled} instead.")
# #                     print("\n\n")
# #                     new_buffer = new_buffer[:n_tokens_filled, ...]
# #                     return new_buffer

# #                 activations = torch.load(
# #                     f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt"
# #                 )
# #                 taking_subset_of_file = False
# #                 if n_tokens_filled + activations.shape[0] > buffer_size:
# #                     activations = activations[: buffer_size - n_tokens_filled, ...]
# #                     taking_subset_of_file = True

# #                 new_buffer[
# #                     n_tokens_filled : n_tokens_filled + activations.shape[0], ...
# #                 ] = activations

# #                 if taking_subset_of_file:
# #                     self.next_idx_within_buffer = activations.shape[0]
# #                 else:
# #                     self.next_cache_idx += 1
# #                     self.next_idx_within_buffer = 0

# #                 n_tokens_filled += activations.shape[0]

# #             return new_buffer

# #         refill_iterator = range(0, batch_size * n_batches_in_buffer, batch_size)
# #         # Initialize empty tensor buffer of the maximum required size with an additional dimension for layers
# #         new_buffer = torch.zeros(
# #             (total_size, context_size, num_layers, d_in),
# #             dtype=self.cfg.dtype,
# #             device=self.cfg.device,
# #         )

# #         for refill_batch_idx_start in refill_iterator:
# #             refill_batch_tokens = self.get_batch_tokens()
# #             refill_activations = self.get_activations(refill_batch_tokens)
# #             new_buffer[
# #                 refill_batch_idx_start : refill_batch_idx_start + batch_size, ...
# #             ] = refill_activations

# #             # pbar.update(1)

# #         new_buffer = new_buffer.reshape(-1, num_layers, d_in)
# #         new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]

# #         return new_buffer

# #     def get_data_loader(
# #         self,
# #     ) -> Iterator[Any]:
# #         """
# #         Return a torch.utils.dataloader which you can get batches from.

# #         Should automatically refill the buffer when it gets to n % full.
# #         (better mixing if you refill and shuffle regularly).
# #         """

# #         batch_size = self.cfg.train_batch_size

# #         # 1. # create new buffer by mixing stored and new buffer
# #         mixing_buffer = torch.cat(
# #             [self.get_buffer(self.cfg.n_batches_in_buffer // 2), self.storage_buffer],
# #             dim=0,
# #         )

# #         mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

# #         # 2.  put 50 % in storage
# #         self.storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

# #         # 3. put other 50 % in a dataloader
# #         dataloader = iter(
# #             DataLoader(
# #                 # TODO: seems like a typing bug?
# #                 cast(Any, mixing_buffer[mixing_buffer.shape[0] // 2 :]),
# #                 batch_size=batch_size,
# #                 shuffle=True,
# #             )
# #         )

# #         return dataloader

# #     def next_batch(self):
# #         """
# #         Get the next batch from the current DataLoader.
# #         If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
# #         """
# #         try:
# #             # Try to get the next batch
# #             return next(self.dataloader)
# #         except StopIteration:
# #             # If the DataLoader is exhausted, create a new one
# #             self.dataloader = self.get_data_loader()
# #             return next(self.dataloader)






# ############################################################################################################
# ####### neel nanda's buffer idea for storing activations and using them for training the autoencoder ######
# ############################################################################################################

# # def shuffle_data(all_tokens):
# #     print("Shuffled data")
# #     return all_tokens[torch.randperm(all_tokens.shape[0])]

# # loading_data_first_time = False
# # if loading_data_first_time:
# #     data = load_dataset("NeelNanda/c4-code-tokenized-2b", split="train", cache_dir="/workspace/cache/")
# #     data.save_to_disk("/workspace/data/c4_code_tokenized_2b.hf")
# #     data.set_format(type="torch", columns=["tokens"])
# #     all_tokens = data["tokens"]
# #     all_tokens.shape

# #     all_tokens_reshaped = einops.rearrange(all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128)
# #     all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
# #     all_tokens_reshaped = all_tokens_reshaped[torch.randperm(all_tokens_reshaped.shape[0])]
# #     torch.save(all_tokens_reshaped, "/workspace/data/c4_code_2b_tokens_reshaped.pt")
# # else:
# #     # data = datasets.load_from_disk("/workspace/data/c4_code_tokenized_2b.hf")
# #     all_tokens = torch.load("/workspace/data/c4_code_2b_tokens_reshaped.pt")
# #     all_tokens = shuffle_data(all_tokens)



# # class Buffer():
# #     """
# #     This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. 
# #     It'll automatically run the model to generate more when it gets halfway empty. 
# #     requires a cfg dictionary with the following
# #     buffer_size: int, the size of the buffer
# #     act_size: int, the size of the activations
# #     device: torch device, where to store the buffer
# #     buffer_batches: int, how many batches to run to fill the buffer
# #     model_batch_size: int, how many tokens to run at once
# #     layer: int, which layer to stop at
# #     act_name: str, the name of the activation to store
# #     batch_size: int, how many activations to return at once
# #     """

# #     def __init__(self, cfg):
# #         self.buffer = torch.zeros((cfg["buffer_size"], cfg["act_size"]), dtype=torch.bfloat16, requires_grad=False).to(cfg["device"])
# #         self.cfg = cfg
# #         self.token_pointer = 0
# #         self.first = True
# #         self.refresh()
    
# #     @torch.no_grad()
# #     def refresh(self):
# #         self.pointer = 0
# #         with torch.autocast("cuda", torch.bfloat16):
# #             if self.first:
# #                 num_batches = self.cfg["buffer_batches"]
# #             else:
# #                 num_batches = self.cfg["buffer_batches"]//2
# #             self.first = False
# #             for _ in range(0, num_batches, self.cfg["model_batch_size"]):
# #                 tokens = all_tokens[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]
# #                 _, cache = model.run_with_cache(tokens, stop_at_layer=cfg["layer"]+1, names_filter=cfg["act_name"])
# #                 acts = cache[cfg["act_name"]].reshape(-1, self.cfg["act_size"])
                
# #                 # print(tokens.shape, acts.shape, self.pointer, self.token_pointer)
# #                 self.buffer[self.pointer: self.pointer+acts.shape[0]] = acts
# #                 self.pointer += acts.shape[0]
# #                 self.token_pointer += self.cfg["model_batch_size"]
# #                 # if self.token_pointer > all_tokens.shape[0] - self.cfg["model_batch_size"]:
# #                 #     self.token_pointer = 0

# #         self.pointer = 0
# #         self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(cfg["device"])]

# #     @torch.no_grad()
# #     def next(self):
# #         out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
# #         self.pointer += self.cfg["batch_size"]
# #         if self.pointer > self.buffer.shape[0]//2 - self.cfg["batch_size"]:
# #             # print("Refreshing the buffer!")
# #             self.refresh()
# #         return out