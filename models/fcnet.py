from torch import nn

class FCNet(nn.Module):
    def __init__(
        self, args#feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True
    ):
        super(FCNet, self).__init__()
        self.is_deconv = True# is_deconv
        self.in_channels = 3# in_channels
        self.is_batchnorm = True#is_batchnorm
        self.feature_scale = 4#feature_scale

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 11, 1, 5),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )

    def forward(self, inputs):
        return self.conv(inputs)
