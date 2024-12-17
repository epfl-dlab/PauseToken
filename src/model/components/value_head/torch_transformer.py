import torch
import hydra

class TransformerValueHead(torch.nn.Module):
    def __init__(
            self, transformer_config, hidden_dim, value_head_path: str = None):
        super(TransformerValueHead, self).__init__()
        self.model = hydra.utils.instantiate(transformer_config)

        self.linear_head = torch.nn.Linear(hidden_dim, 1)

        self.query_embedding = torch.nn.Parameter(torch.randn(hidden_dim))
        if value_head_path is not None:
            self.model.load_state_dict(torch.load(value_head_path))
        

    def forward(self, all_hidden_embeds):
        last_layer_hidden_embeds = all_hidden_embeds[-1]
        query_appended_embed = torch.cat([last_layer_hidden_embeds, 
                                          self.query_embedding.unsqueeze(0).expand(last_layer_hidden_embeds.size(0), -1).unsqueeze(1)],
                                            dim=1)
        output = self.model(query_appended_embed)[:, -1, :]
        value = self.linear_head(output)
        return value
    

        