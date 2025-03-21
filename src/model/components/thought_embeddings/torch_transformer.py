import torch
import hydra

class ThoughtTransformer(torch.nn.Module):
    def __init__(
            self, transformer_config, hidden_dim, divide_by_sqrt_dim=False):
        super(ThoughtTransformer, self).__init__()
        if isinstance(transformer_config, torch.nn.Module):
            self.model = transformer_config
        else:
            self.model = hydra.utils.instantiate(transformer_config)
        self.hidden_dim = torch.tensor(hidden_dim, requires_grad=False)
        self.divide_by_sqrt_dim = divide_by_sqrt_dim


    def forward(self, last_hidden_states: torch.Tensor, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones((last_hidden_states.size(0), last_hidden_states.size(1)), device=last_hidden_states.device)
        
        divide = torch.sqrt(self.hidden_dim) if self.divide_by_sqrt_dim else self.hidden_dim        
        thoughts = self.model(inputs_embeds=last_hidden_states, attention_mask=attention_mask).last_hidden_state / divide
                
        return thoughts
    

        