######
# TAKEN FROM https://github.com/teticio/llama-squad/blob/main/llama_squad.py#L49
######

import torch

class ExtendedEmbedding(torch.nn.Module):
    def __init__(
        self, original_embedding: torch.nn.Embedding, new_embedding: torch.nn.Embedding
    ):
        super(ExtendedEmbedding, self).__init__()
        self.original_embedding = original_embedding
        self.new_embedding = new_embedding
        self.num_og_embeddings = original_embedding.num_embeddings

    def forward(self, input_ids):
        is_new_token = input_ids >= self.num_og_embeddings
        original_tokens = input_ids[~is_new_token]
        original_embeddings = self.original_embedding(original_tokens)

        combined_embeddings = (
            torch.zeros(input_ids.shape + (original_embeddings.shape[1],))
            .to(original_embeddings.device)
            .to(original_embeddings.dtype)
        )
        combined_embeddings[~is_new_token] = original_embeddings

        new_tokens = input_ids[is_new_token] - self.num_og_embeddings
        if len(new_tokens) > 0:
            combined_embeddings[is_new_token] = self.new_embedding(new_tokens).to(
                original_embeddings.device
            )

        return combined_embeddings