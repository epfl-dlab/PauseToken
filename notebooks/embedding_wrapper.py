from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import transformers
from tokenizers import AddedToken
import torch
from torch import nn

class EmbeddingWrapperMask(nn.Module):
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
        old_x_mask = x < self.num_old_embeddings
        new_x_mask = x >= self.num_old_embeddings

        old_x = x[old_x_mask]
        new_x = x[new_x_mask] - self.num_old_embeddings

        output = torch.zeros((len(x), self.old_embedding.embedding_dim), dtype=self.old_embedding.weight.dtype, device=self.old_embedding.weight.device)
        output[old_x_mask] = self.old_embedding(old_x)
        output[new_x_mask] = self.new_embedding(new_x)

        return output

class EmbeddingWrapperHook(nn.Module):
    def __init__(self, old_embedding: nn.Embedding, num_embeddings: int, freeze_old=True):
        super().__init__()
        self.old_embedding = nn.Embedding(old_embedding.num_embeddings, old_embedding.embedding_dim)
        self.embedding_dim = old_embedding.embedding_dim
        self.old_embedding.weight.data = old_embedding.weight.data.clone()
        self.new_embedding = nn.Embedding(num_embeddings - old_embedding.num_embeddings, self.embedding_dim)

        self.num_old_embeddings = old_embedding.num_embeddings

        if freeze_old:
            self.old_embedding.weight.register_hook(lambda grad: grad * 0)

    def forward(self, x):
        old_x = x[x < self.num_old_embeddings]
        new_x = x[x >= self.num_old_embeddings] - self.num_old_embeddings
        return torch.cat([self.old_embedding(old_x), self.new_embedding(new_x)], dim=0)


class PartiallyFrozenEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, old_embedding_weight, new_embedding_weight, num_old_embeddings, embedding_dim, x):
        old_x_mask = x < num_old_embeddings
        new_x_mask = x >= num_old_embeddings

        old_x = x[old_x_mask]
        new_x = x[new_x_mask] - num_old_embeddings

        # set up an output vector, same length as x and same embedding dimension
        num_outputs = x.shape[0]
        output = torch.empty((num_outputs, embedding_dim), dtype=old_embedding_weight.dtype, device=old_embedding_weight.device)

        output[old_x_mask] = torch.nn.functional.embedding(old_x, old_embedding_weight)
        output[new_x_mask] = torch.nn.functional.embedding(new_x, new_embedding_weight)

        ctx.save_for_backward(old_embedding_weight, new_embedding_weight, old_x, new_x, old_x_mask, new_x_mask)
        ctx.num_old_embeddings = num_old_embeddings

        return output

    @staticmethod
    def backward(ctx, grad_output):
        old_embedding_weight, new_embedding_weight, old_x, new_x, old_x_mask, new_x_mask = ctx.saved_tensors

        grad_old_embedding = torch.zeros_like(old_embedding_weight)
        grad_new_embedding = torch.zeros_like(new_embedding_weight)

        grad_old_embedding.index_add_(0, old_x, grad_output[old_x_mask])
        grad_new_embedding.index_add_(0, new_x, grad_output[new_x_mask])

        return None, grad_new_embedding, None, None, None, None

class EmbeddingWrapperFunction(nn.Module):
    def __init__(self, old_embedding: nn.Embedding, num_embeddings: int):
        super().__init__()
        self.old_embedding = nn.Embedding(old_embedding.num_embeddings, old_embedding.embedding_dim)
        self.embedding_dim = old_embedding.embedding_dim
        self.old_embedding.weight.data = old_embedding.weight.data.clone()
        self.new_embedding = nn.Embedding(num_embeddings - old_embedding.num_embeddings, self.embedding_dim)
        self.num_old_embeddings = old_embedding.num_embeddings

        self.old_embedding.weight.requires_grad = True
        self.new_embedding.weight.requires_grad = True

    def forward(self, x):
        return PartiallyFrozenEmbedding.apply(self.old_embedding.weight, self.new_embedding.weight, self.num_old_embeddings, self.embedding_dim,  x)

if __name__=="__main__":
    import torch
    from torch import nn
    from torch.optim import SGD

    # Step 1: Create an instance of nn.Embedding as the old embedding and initialize its weights randomly.
    old_embedding = nn.Embedding(10, 32)
    old_embedding.weight.data.normal_()

    # Step 2: Create an instance of EmbeddingWrapper3, passing the old embedding and the desired number of embeddings to its constructor.
    num_embeddings = 15
    embedding_wrapper = EmbeddingWrapperHook(old_embedding, num_embeddings)

    # Step 3: Create a linear layer on top of the EmbeddingWrapper3.
    linear_layer = nn.Linear(embedding_wrapper.embedding_dim, 1)

    # Step 4: Create some synthetic training data.
    x = torch.randint(num_embeddings, (100,))  # 100 random integers between 0 and num_embeddings
    y = torch.randint(2, (100,)).float()  # 100 random 0s and 1s

    # Step 5: Train the model using a simple training loop.
    optimizer = SGD(list(embedding_wrapper.parameters()) + list(linear_layer.parameters()), lr=0.5)
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
    print("test passed: ", torch.allclose(old_embedding_weights, embedding_wrapper.old_embedding.weight.data))

    # check that the new embedding weights are different
    print("test passed: ", not torch.allclose(new_embedding_weights, embedding_wrapper.new_embedding.weight.data))

    print(old_embedding_weights[0,0], embedding_wrapper.old_embedding.weight.data[0,0])
    print(new_embedding_weights[0,0], embedding_wrapper.new_embedding.weight.data[0,0])