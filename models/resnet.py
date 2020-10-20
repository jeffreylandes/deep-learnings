from layers import CNNBlock, ResBlock
from torch.nn import Sequential, Module, MaxPool2d, Conv2d, ReLU


class ResNet(Module):

    def __init__(self, **kwargs):
        super(ResNet, self).__init__()

        num_input_channels = kwargs.get("num_input_channels", 3)
        num_out_channels = kwargs.get("num_out_channels", 1)

        self.first_conv_layer = CNNBlock(num_input_channels, 4, 3, 1, 1)

        self.low_conv_down = Sequential(
            ResBlock(4, 16, 5, 1, 2),
            MaxPool2d(2, 2),
            ResBlock(16, 32, 5, 1, 2),
            MaxPool2d(2, 2),
            ResBlock(32, 64, 3, 1, 1),
            MaxPool2d(2, 2),
            ResBlock(64, 128, 3, 1, 1),
            ResBlock(128, 128, 3, 1, 1),
        )

        self.res_combine1 = ResBlock(20, 8, 3, 1, 1)

        self.res_combine2 = ResBlock(8, 8, 3, 1, 1)

        self.res_final = ResBlock(num_input_channels + 8 + 1, 8, 3, 1, 1)

        self.conv_final = Sequential(Conv2d(8, num_out_channels, 3, 1, 1), ReLU())

    def forward(self, x, mask):

        x = self.first_conv_layer_high(x)

        x = self.low_conv_down(x)
        return x_final