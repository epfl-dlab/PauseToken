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


class EmbeddingWrapper3(nn.Module):
    def __init__(self, old_embedding: nn.Embedding, num_embeddings: int, freeze_old=True):
        super().__init__()
        self.old_embedding = nn.Embedding(old_embedding.num_embeddings, old_embedding.embedding_dim)
        self.embedding_dim = old_embedding.embedding_dim
        self.old_embedding.weight.data = old_embedding.weight.data.clone()
        self.new_embedding = nn.Embedding(num_embeddings - old_embedding.num_embeddings, self.embedding_dim)

        if freeze_old:
            for param in self.old_embedding.parameters():
                param.requires_grad = False

        self.num_old_embeddings = old_embedding.num_embeddings

    def forward(self, x):
        old_x = x[x < self.num_old_embeddings]
        new_x = x[x >= self.num_old_embeddings] - self.num_old_embeddings
        return torch.cat([self.old_embedding(old_x), self.new_embedding(new_x)], dim=0)


if __name__=="__main__":
    import torch
    from torch import nn
    from torch.optim import SGD

    # Step 1: Create an instance of nn.Embedding as the old embedding and initialize its weights randomly.
    old_embedding = nn.Embedding(10, 32)
    old_embedding.weight.data.normal_()

    # Step 2: Create an instance of EmbeddingWrapper3, passing the old embedding and the desired number of embeddings to its constructor.
    num_embeddings = 15
    embedding_wrapper = EmbeddingWrapper3(old_embedding, num_embeddings)

    # Step 3: Create a linear layer on top of the EmbeddingWrapper3.
    linear_layer = nn.Linear(embedding_wrapper.embedding_dim, 1)

    # Step 4: Create some synthetic training data.
    x = torch.randint(num_embeddings, (100,))  # 100 random integers between 0 and num_embeddings
    y = torch.randint(2, (100,)).float()  # 100 random 0s and 1s

    # Step 5: Train the model using a simple training loop.
    optimizer = SGD(list(embedding_wrapper.parameters()) + list(linear_layer.parameters()), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # store the old embedding weights
    old_embedding_weights = old_embedding.weight.data.clone()
    new_embedding_weights = embedding_wrapper.new_embedding.weight.data.clone()

    for epoch in range(10):
        optimizer.zero_grad()
        embeddings = embedding_wrapper(x)
        logits = linear_layer(embeddings).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # check that the old embedding weights are the same
    print("test passed: ", torch.allclose(old_embedding_weights, old_embedding.weight.data))

    # check that the new embedding weights are different
    print("test passed: ", not torch.allclose(new_embedding_weights, embedding_wrapper.new_embedding.weight.data))