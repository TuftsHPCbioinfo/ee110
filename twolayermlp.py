import torch.nn as nn

class TwoLayerMLP(nn.Module):
    """
    Two-layer MLP as described in Experiment 2 of "Adam: A Method for Stochastic Optimization"
    Fill in all TODOs to complete the network.
    """
    def __init__(self,
                 input_size: int = 784,
                 hidden_size: int = 1024,
                 num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            # TODO: Linear layer (input_size -> hidden_size)
            nn.Linear(input_size, hidden_size),
            # TODO: Activation
            nn.ReLU(),
            # TODO: Linear layer (hidden_size -> hidden_size)
            nn.Linear(hidden_size, hidden_size),
            # TODO: Activation
            nn.ReLU(),
            # TODO: Output linear layer (hidden_size -> num_classes)
            nn.Linear(hidden_size, num_classes),
        )


    def forward(self, x):
        out = self.net(x)
        return out