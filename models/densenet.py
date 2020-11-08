from torch.nn import Conv2d, Sequential, Module, ReLU, ModuleList, MaxPool2d, Linear
from models.layers import CNNBlock
from torch import cat


class DenseBlock(Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_out_channels,
                 kernel,
                 stride,
                 padding,
                 pooling,
                 relu=ReLU(),
                 k=4):
        super(DenseBlock, self).__init__()

        self.layers = ModuleList([
            CNNBlock(in_channels=in_channels + (i * out_channels),
                     out_channels=out_channels,
                     kernel=kernel,
                     stride=stride,
                     padding=padding,
                     relu=relu)
            for i in range(k)
        ])

        self.convolution_pooling = Sequential(
            Conv2d(out_channels, final_out_channels, kernel, padding),
            MaxPool2d(pooling, pooling)
        )

    def forward(self, x):
        x_cat = x
        for i in range(len(self.layers)):
            x_new = self.layers[i](x_cat)
            x_cat = cat([x_cat, x_new], dim=1)

        output = self.convolution_pooling(x_new)
        return output


class DenseNet(Module):

    def __init__(self):
        super(DenseNet, self).__init__()

        self.initial_convolution = Sequential(
            Conv2d(3, 8, 3, 1, 1),
            ReLU()
        )

        self.dense_block_1 = DenseBlock(
            in_channels=8,
            out_channels=8,
            final_out_channels=16,
            kernel=3,
            stride=1,
            padding=1,
            pooling=3
        )

        self.dense_block_2 = DenseBlock(
            in_channels=16,
            out_channels=16,
            final_out_channels=32,
            kernel=3,
            stride=1,
            padding=1,
            pooling=3
        )

        self.dense_block_3 = DenseBlock(
            in_channels=32,
            out_channels=32,
            final_out_channels=64,
            kernel=3,
            stride=1,
            padding=1,
            pooling=2
        )

        self.linear = Sequential(
            Linear(9 * 9 * 64, 64),
            ReLU(),
            Linear(64, 8),
            ReLU(),
            Linear(8, 1)
        )

    def forward(self, x):
        x = self.initial_convolution(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        x = self.dense_block_3(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        # Think we can return this and do cross-entropy with logits
        return x
