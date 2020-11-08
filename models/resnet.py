from models.layers import CNNBlock, ResBlock
from torch.nn import Sequential, Module, MaxPool2d, Linear, ReLU


class ResNet(Module):

    def __init__(self, **kwargs):
        super(ResNet, self).__init__()

        num_input_channels = kwargs.get("num_input_channels", 3)

        self.first_convolutional_layer = CNNBlock(num_input_channels, 4, 3, 1, 1)

        self.residual_layers = Sequential(
            ResBlock(4, 16, 5, 1, 2),
            MaxPool2d(2, 2),
            ResBlock(16, 32, 5, 1, 2),
            MaxPool2d(2, 2),
            ResBlock(32, 64, 3, 1, 1),
            MaxPool2d(2, 2),
            ResBlock(64, 128, 3, 1, 1),
            MaxPool2d(2, 2),
            ResBlock(128, 128, 3, 1, 1),
            MaxPool2d(2, 2)
        )

        self.linear = Sequential(
            Linear(4608, 64),
            ReLU(),
            Linear(64, 8),
            ReLU(),
            Linear(8, 1)
        )

    def forward(self, x):

        x = self.first_convolutional_layer(x)
        x = self.residual_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)

        return x
