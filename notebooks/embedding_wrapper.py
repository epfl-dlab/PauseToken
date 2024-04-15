from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import transformers
from tokenizers import AddedToken
import torch
from torch import nn


# write torch.nn.Modules that can be used as a wrapper around huggingface models and change the embedding layer
# and the unembedding layer
class LinearWrapper(nn.Module):
    def __init__(self, layer: nn.Linear, num_embeddings: int, freeze_old=True):
        super().__init__()
        self.layer = layer
        self.num_embeddings = num_embeddings
        self.n_new_tokens = num_embeddings - layer.out_features
        self.new_embeddings = nn.Linear(layer.in_features, self.n_new_tokens, bias=False)
        self.new_embeddings.to(layer.weight.device).to(layer.weight.dtype)
        if freeze_old:
            for param in self.layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        z1 = self.layer(x)
        z2 = self.new_embeddings(x)
        return torch.cat([z1, z2], dim=-1)


class EmbeddingWrapper(nn.Module):
    def __init__(self, embedding: nn.Embedding, num_embeddings: int, freeze_old=True):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.num_embeddings = num_embeddings
        self.n_new_tokens = num_embeddings - embedding.num_embeddings

        # inspired from here
        # https://github.com/huggingface/transformers/blob/185463784e0a0b4cd7974ce5bded7a52ae170f6d/src/transformers/modeling_utils.py#L2026
        self.old_embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.old_embeddings.weight.data = torch.ones_like(self.old_embeddings.weight.data) * 0  # 1e-7
        self.old_embeddings.weight.data[:embedding.num_embeddings] = embedding.weight.data
        self.old_embeddings.to(embedding.weight.device).to(embedding.weight.dtype)
        self.new_embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.new_embeddings.weight.data[:embedding.num_embeddings] = torch.ones_like(embedding.weight.data) * 0  # 1e-7
        self.new_embeddings.to(embedding.weight.device).to(embedding.weight.dtype)
        if freeze_old:
            for param in self.old_embeddings.parameters():
                param.requires_grad = False

    def forward(self, x):
        self.old_embeddings(x) + self.new_embeddings(x)


class EmbeddingWrapper2(nn.Module):
    def __init__(self, embedding: nn.Embedding, num_embeddings: int, freeze_old=True):
        super().__init__()
        self.old_embeddings = embedding
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.n_new_tokens = num_embeddings - embedding.num_embeddings
        self.new_embeddings = nn.Embedding(self.n_new_tokens, self.embedding_dim)
        self.new_embeddings.to(embedding.weight.device).to(embedding.weight.dtype)
        if freeze_old:
            for param in self.old_embeddings.parameters():
                param.requires_grad = False

    def forward(self, x):
        with torch.amp.autocast("cuda"):
            mask_small = x < self.old_embeddings.num_embeddings
            mask_large = x >= self.old_embeddings.num_embeddings
            small_ids = x[mask_small]
            large_ids = x[mask_large]
            small_embs = self.old_embeddings(small_ids)
            large_embs = self.new_embeddings(large_ids % self.old_embeddings.num_embeddings)
            # assuming batch x seq x emb
            y = torch.empty((x.shape[0], x.shape[1], small_embs.shape[-1]), dtype=large_embs.dtype,
                            device=large_embs.device)
            y[mask_small] = small_embs
            y[mask_large] = large_embs
            return y


class Llama2EmbeddingSurgeon():
    def __init__(self, llama, extended_tokenizer):
        self.llama = llama
        self.extended_tokenizer = extended_tokenizer
        self.extended_embedding = EmbeddingWrapper2(llama.model.embed_tokens, len(extended_tokenizer))
        self.extended_unembedding = LinearWrapper(llama.lm_head, len(extended_tokenizer))

    def get_surgeried_model(self):
        self.backup_embed_tokens = self.llama.model.embed_tokens
        self.backup_lm_head = self.llama.lm_head
        self.llama.model.embed_tokens = self.extended_embedding
        self.llama.lm_head = self.extended_unembedding
        self.llama.config.vocab_size = len(self.extended_tokenizer)
        return self.llama

    def save(self, llama, path):
        # check if llama is surgeried
        assert llama.model.embed_tokens == self.extended_embedding
        assert llama.lm_head == self.extended_unembedding
        backup_embed_tokens = self.llama.model.embed_tokens
        backup_lm_head = self.llama.lm_head
        self.llama.model.embed_tokens = self.backup_embed_tokens
        self.llama.lm_head = self.backup_lm_head
        self.llama.save_pretrained(path)
        self.llama.model.embed_tokens = backup_embed_tokens
        self.llama.lm_head = backup_lm_head
        self.extended_tokenizer.save_pretrained(path)
        torch.save(self.extended_embedding.state_dict(), f"{path}/extended_embedding.pt")
        torch.save(self.extended_unembedding.state_dict(), f"{path}/extended_unembedding.pt")

    @classmethod
    def load(cls, path):
        extended_embedding_dict = torch.load(f"{path}/extended_embedding.pt")
        extended_unembedding_dict = torch.load(f"{path}/extended_unembedding.pt")
        llama = AutoModelForCausalLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        surgeon = cls(llama, tokenizer)
        surgeon.extended_embedding.load_state_dict(extended_embedding_dict)
        surgeon.extended_unembedding.load_state_dict(extended_unembedding_dict)
        return surgeon


class PeftModelEmbeddingSurgeon():
    def __init__(self, peft_model, extended_tokenizer):
        try:
            self.llama = peft_model.base_model.model
        except AttributeError:
            self.llama = peft_model
        self.peft_model = peft_model
        self.extended_tokenizer = extended_tokenizer
        self.extended_embedding = EmbeddingWrapper2(self.llama.model.embed_tokens, len(extended_tokenizer))
        self.extended_unembedding = LinearWrapper(self.llama.lm_head, len(extended_tokenizer))

    def get_surgeried_model(self):
        self.backup_embed_tokens = self.llama.model.embed_tokens
        self.backup_lm_head = self.llama.lm_head
        self.llama.model.embed_tokens = self.extended_embedding
        self.llama.lm_head = self.extended_unembedding
        self.llama.config.vocab_size = len(self.extended_tokenizer)
        return self.peft_model

    def save(self, peft_model, path):
        self.llama.model.embed_tokens = self.backup_embed_tokens
        self.llama.lm_head = self.backup_lm_head
        self.peft_model.save_pretrained(path)
        self.extended_tokenizer.save_pretrained(path)
        torch.save(self.extended_embedding.state_dict(), f"{path}/extended_embedding.pt")
        torch.save(self.extended_unembedding.state_dict(), f"{path}/extended_unembedding.pt")

    @classmethod
    def load(cls, path, **kwargs):
        extended_embedding_dict = torch.load(f"{path}/extended_embedding.pt")
        extended_unembedding_dict = torch.load(f"{path}/extended_unembedding.pt")
        peft_model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(path)
        surgeon = cls(peft_model, tokenizer)
        surgeon.extended_embedding.load_state_dict(extended_embedding_dict)
        surgeon.extended_unembedding.load_state_dict(extended_unembedding_dict)
        return surgeon