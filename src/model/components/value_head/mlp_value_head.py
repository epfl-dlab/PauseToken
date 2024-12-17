import torch

class MLPValueHead(torch.nn.Module):
    def __init__(
            self, hidden_sizes: list, activation: str = 'relu', output_activation: str = 'none'
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
        

    def forward(self, input_ids):
        return self.model(input_ids)
    

        