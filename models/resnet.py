from layers import CNNBlock, ResBlock


class ResNet(Module):
    """
    Combines large kernels (low frequency) with residual layers (high frequency)
    to compute the wind vector field of a site.
    Magnitude of the vector.
    """

    def __init__(self, **kwargs):
        super(ResNet, self).__init__()

        num_input_channels = kwargs.get("num_input_channels", 3)
        num_out_channels = kwargs.get("num_out_channels", 1)

        # can change
        num_deep_channels_high = kwargs.get("num_deep_channels_high", 12)
        num_deep_channels_low = kwargs.get("num_deep_channels_low", 12)

        self.first_conv_layer_low = CNNBlock(num_input_channels, 4, 3, 1, 1)

        self.first_conv_layer_high = CNNBlock(
            num_input_channels, num_deep_channels_high, 3, 1, 1
        )

        # Large kernels capture low frequency features (Long wind tunnels, etc.)
        # TODO: Show how this can be simplified by defining a block
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

        self.low_conv_up_1 = CNNBlock(128, 64, 3, 1, 1)
        self.low_conv_up_2 = CNNBlock(64, 32, 3, 1, 1)
        self.low_conv_up_3 = CNNBlock(32, 8, 3, 1, 1)

        self.low_conv_final_1 = Sequential(
            Conv2d(num_deep_channels_low, num_deep_channels_low, 3, 1, 1),
            BatchNorm2d(num_deep_channels_low),
            ReLU(),
        )
        self.low_conv_final_2 = Sequential(
            Conv2d(num_deep_channels_low, num_deep_channels_low, 3, 1, 1),
            BatchNorm2d(num_deep_channels_low),
            ReLU(),
        )
        self.low_conv_final_3 = Sequential(
            Conv2d(num_deep_channels_low, num_deep_channels_low, 3, 1, 1),
            BatchNorm2d(num_deep_channels_low),
            ReLU(),
        )

        # Residual layers capture high frequency features (wind close to building edges, corners, etc.)
        self.res1 = ResBlock(num_deep_channels_high, num_deep_channels_high, 5, 1, 2)
        self.res2 = ResBlock(num_deep_channels_high, num_deep_channels_high, 5, 1, 2)
        self.res3 = ResBlock(num_deep_channels_high, num_deep_channels_high, 3, 1, 1)
        self.res4 = ResBlock(num_deep_channels_high, num_deep_channels_high, 3, 1, 1)
        self.res5 = ResBlock(num_deep_channels_high, num_deep_channels_high, 3, 1, 1)

        self.res_combine1 = ResBlock(20, 8, 3, 1, 1)

        self.res_combine2 = ResBlock(8, 8, 3, 1, 1)

        self.res_final = ResBlock(num_input_channels + 8 + 1, 8, 3, 1, 1)

        self.conv_final = Sequential(Conv2d(8, num_out_channels, 3, 1, 1), ReLU())

    def forward(self, x, mask):

        x_high = self.first_conv_layer_high(x)
        x_low = self.first_conv_layer_low(x)

        # Capture low frequency features (large wind tunnels, open spaces, etc.)
        x_low = self.low_conv_down(x_low)
        x_low = functional.interpolate(x_low, scale_factor=(2, 2), mode="nearest")
        x_low = self.low_conv_up_1(x_low)
        x_low = functional.interpolate(x_low, scale_factor=(2, 2), mode="nearest")
        x_low = self.low_conv_up_2(x_low)
        x_low = functional.interpolate(x_low, scale_factor=(2, 2), mode="nearest")
        x_low = self.low_conv_up_3(x_low)

        # Capture high frequency features (building edges, compact wind tunnels, etc.)
        x_high = self.res1(x_high)
        x_high = self.res2(x_high)
        x_high = self.res3(x_high)
        x_high = self.res4(x_high)
        x_high = self.res5(x_high)

        # Will have 12 channels --> compute a couple more residual layers
        x_concat = torch.cat((x_low, x_high), dim=1)

        x_combined = self.res_combine1(x_concat)
        x_combined = self.res_combine2(x_combined)

        # One last concatenation
        x_mask = torch.cat((x_combined, x, mask), dim=1)
        x_final = self.res_final(x_mask)
        x_final = self.conv_final(x_final)
        return x_final