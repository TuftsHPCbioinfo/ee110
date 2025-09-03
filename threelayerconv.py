import torch.nn as nn

class ThreeLayerConv(nn.Module):
    """
    Three-layer CNN as described in Experiment 3 of 
    "Adam: A Method for Stochastic Optimization":

    Our CNN architecture has three alternating stages of 5x5
    convolution filters and 3x3 max pooling with stride of 2 
    that are followed by a fully connected layer of 1000 ReLU units.
    """
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # TODO: First conv layer (input_channels -> 64), 5x5 kernel, padding=2
            nn.Conv2d(input_channels, 64, kernel_size=5, padding=2),  # 32x32 -> 32x32
            # TODO: ReLU activation
            nn.ReLU(),
            # TODO: First max pooling layer (3x3 kernel, stride=2)
            nn.MaxPool2d(kernel_size=3, stride=2),                    # 32x32 -> 15x15
            # TODO: Second conv layer (64 -> 64), 5x5 kernel, padding=2
            nn.Conv2d(64, 64, kernel_size=5, padding=2),              # 15x15 -> 15x15
            # TODO: ReLU activation
            nn.ReLU(),
            # TODO: Second max pooling layer (3x3 kernel, stride=2)
            nn.MaxPool2d(kernel_size=3, stride=2),                    # 15x15 -> 7x7
            # TODO: Third conv layer (64 -> 128), 5x5 kernel, padding=2
            nn.Conv2d(64, 128, kernel_size=5, padding=2),             # 7x7 -> 7x7
            # TODO: ReLU activation
            nn.ReLU(),
            # TODO: Third max pooling layer (3x3 kernel, stride=2)
            nn.MaxPool2d(kernel_size=3, stride=2),                    # 7x7 -> 3x3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
