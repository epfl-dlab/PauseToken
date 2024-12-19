import torch

class MLPValueHead(torch.nn.Module):
    def __init__(
            self, hidden_sizes: list, activation: str = 'relu', output_activation: str = 'none', value_head_path: str = None
    ):
        super(MLPValueHead, self).__init__()
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

        if value_head_path is not None:
            self.load_state_dict(torch.load(value_head_path, weights_only=True))
        

    def forward(self, all_hidden_embeds, attention_mask=None):
        last_layer_hidden_embeds = all_hidden_embeds[-1]
        last_token_hidden_embed = last_layer_hidden_embeds[:, -1]
        return self.model(last_token_hidden_embed)
    

        