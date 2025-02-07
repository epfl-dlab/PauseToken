import torch

class MLPThought(torch.nn.Module):
    def __init__(
            self, hidden_sizes: list, activation: str = 'relu', output_activation: str = 'none'
    ):
        super(MLPThought, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.layers = []
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i < len(hidden_sizes) - 2:
                if activation == 'relu':
                    self.layers.append(torch.nn.ReLU())
                elif activation == 'tanh':
                    self.layers.append(torch.nn.Tanh())
                elif activation == 'sigmoid':
                    self.layers.append(torch.nn.Sigmoid())
                else:
                    raise ValueError('Invalid activation function')
        if output_activation == 'relu':
            self.layers.append(torch.nn.ReLU())
        elif output_activation == 'tanh':
            self.layers.append(torch.nn.Tanh())
        elif output_activation == 'sigmoid':
            self.layers.append(torch.nn.Sigmoid())
        elif output_activation == 'none':
            pass
        else:
            raise ValueError('Invalid output activation function')
        self.model = torch.nn.Sequential(*self.layers)        

    def forward(self, last_hidden_states: torch.Tensor , attention_mask=None):
        #all_hidden_embeds should be of shape (batch_size, seq_len, hidden_dim). So, we need to reshape it to (batch_size * seq_len, hidden_dim)
        reshaped_hidden_embeds = last_hidden_states.view(-1, last_hidden_states.size(-1))

        #after this we should get thoughts of shape (batch_size * seq_len, thought_dim)
        thoughts = self.model(reshaped_hidden_embeds)

        return thoughts.view(last_hidden_states.size(0), last_hidden_states.size(1), -1)
    

        