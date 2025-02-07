import torch
import hydra

class ThoughtTransformer(torch.nn.Module):
    def __init__(
            self, transformer_config, hidden_dim):
        super(ThoughtTransformer, self).__init__()
        self.model = hydra.utils.instantiate(transformer_config)
        self.hidden_dim = hidden_dim


    def forward(self, last_hidden_states: torch.Tensor, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones((last_hidden_states.size(0), last_hidden_states.size(1)), device=last_hidden_states.device)

        thoughts = self.model(inputs_embeds=last_hidden_states, attention_mask=attention_mask).last_hidden_state
        
        return thoughts
    

        