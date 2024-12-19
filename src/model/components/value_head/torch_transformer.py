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
        

    def forward(self, all_hidden_embeds, attention_mask=None):
        last_layer_hidden_embeds = all_hidden_embeds[-1]
        if attention_mask is None:
            attention_mask = torch.ones((last_layer_hidden_embeds.size(0), last_layer_hidden_embeds.size(1)), device=last_layer_hidden_embeds.device)
        query_appended_embed = torch.cat([last_layer_hidden_embeds, 
                                          self.query_embedding.unsqueeze(0).expand(last_layer_hidden_embeds.size(0), -1).unsqueeze(1)],
                                            dim=1)
        
        attention_mask_appended = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)], dim=1)
        attention_mask_appended = attention_mask_appended.unsqueeze(1).expand(-1, attention_mask_appended.size(1), -1)
        sequence_attention_mask = torch.logical_not(attention_mask_appended)

        output = self.model(query_appended_embed, mask=sequence_attention_mask)
        value = self.linear_head(output)
        value = torch.nn.functional.sigmoid(value)
        return value
    

        