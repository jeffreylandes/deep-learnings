from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    LeakyReLU,
    Module,
    ReLU,
    Sequential,
)


class CNNBlock(Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, relu=ReLU):
        super(CNNBlock, self).__init__()

        self.cnn_layer = Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(out_channels),
            relu(),
        )

    def forward(self, X):

        return self.cnn_layer(X)


class ResBlock(Module):
    """
    Single residual layer.
    """

    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super(ResBlock, self).__init__()

        self.conv = CNNBlock(
            in_channels, out_channels, kernel, stride, padding, LeakyReLU
        )
        self.conv_same_1 = CNNBlock(
            out_channels, out_channels, kernel, stride, padding, LeakyReLU
        )
        self.conv_same_2 = CNNBlock(
            out_channels, out_channels, kernel, stride, padding, LeakyReLU
        )
        self.conv_same_3 = CNNBlock(
            out_channels, out_channels, kernel, stride, padding, LeakyReLU
        )
        self.conv_same_4 = CNNBlock(
            out_channels, out_channels, kernel, stride, padding, LeakyReLU
        )
        self.conv_same_5 = CNNBlock(
            out_channels, out_channels, kernel, stride, padding, LeakyReLU
        )
        self.up_dims = Sequential(
            ConvTranspose2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, X):
        x1 = self.conv(X)
        X = self.up_dims(X)
        X = X + x1
        x2 = self.conv_same_1(X)
        X = X + x2
        x3 = self.conv_same_2(X)
        X = X + x3
        x4 = self.conv_same_3(X)
        X = X + x4
        x5 = self.conv_same_4(X)
        X = X + x5
        x6 = self.conv_same_5(X)
        X = X + x6

        return X
