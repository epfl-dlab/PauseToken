import torch
import hydra

class ThoughtTransformer(torch.nn.Module):
    def __init__(
            self, transformer_config, hidden_dim, divisor_exponent):
        super(ThoughtTransformer, self).__init__()
        if isinstance(transformer_config, torch.nn.Module):
            self.model = transformer_config
        else:
            self.model = hydra.utils.instantiate(transformer_config)
        
        #get the input embeddings of the model
        input_embeddings = self.model.get_input_embeddings()    
        
        # WHY? Because in distributed training if this will yield an error. Something like this:
        # "Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss."
        if input_embeddings.weight.shape[0] == 0:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.hidden_dim = torch.tensor(hidden_dim, requires_grad=False)
        self.divisor_exponent =  torch.tensor(divisor_exponent, requires_grad=False)

    def forward(self, last_hidden_states: torch.Tensor, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones((last_hidden_states.size(0), last_hidden_states.size(1)), device=last_hidden_states.device)
        
        divide = self.hidden_dim ** self.divisor_exponent
        thoughts = self.model(inputs_embeds=last_hidden_states, attention_mask=attention_mask).last_hidden_state / divide
                
        return thoughts
    

        