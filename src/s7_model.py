import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class S7_CNNModel(nn.Module):

    def __init__(self):
        super(S7_CNNModel, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1A = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 32x32x3 , out = 32x32x32, RF = 3

        self.depthwise1A = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 32x32x3 * 3x3x3 , out = 32x32x1x32, RF = 5
        self.pointwise1A = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 32x32x1X32 , out = 32x32x64, RF = 5

        self.depthwise1B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 32x32x64 , out = 32x32x64, RF = 7
        self.pointwise1B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 32x32x1X64 , out = 32x32x64, RF = 7

        self.depthwise1C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 32x32x64 , out = 32x32x64, RF = 9
        self.pointwise1C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 32x32x1X64 , out = 32x32x32   , RF = 9

        # TRANSITION BLOCK 1
        self.pool1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 32x32x32 , out = 28x28x32  , RF = 13

        # CONVOLUTION BLOCK 2

        self.convblock2A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 24x24x32 , out = 20x20x32, RF = 15

        self.depthwise2A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 28x28x32 * 3x3x3 , out = 28x28x1x32, RF = 17
        self.pointwise2A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 28x28x1X32 , out = 28x28x64    , RF = 17

        self.depthwise2B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 28x28x1x64 , out = 28x28x64, RF = 19
        self.pointwise2B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 28x28x64 , out = 28x28x64 , RF = 19

        self.depthwise2C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 28x28x1x64 , out = 28x28x64, RF = 21
        self.pointwise2C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 28x28x64 , out = 28x28x32  , RF = 21

        # TRANSITION BLOCK 2
        self.pool2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 28x28x32 , out = 24x24x32  , RF = 25

        # CONVOLUTION BLOCK 3
        self.convblock3A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 24x24x32 , out = 20x20x32, RF = 27

        self.depthwise3A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 20x20x32 * 3x3x3 , out = 20x20x1x32, RF = 29
        self.pointwise3A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 20x20x1X32 , out = 20x20x64, RF = 29

        self.depthwise3B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 20x20x1x64 , out = 20x20x64, RF = 31
        self.pointwise3B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 20x20x64 , out = 20x20x64 , RF = 31

        self.depthwise3C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 20x20x1x64 , out = 20x20x64, RF = 33
        self.pointwise3C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 20x20x64 , out = 20x20x32      , RF = 33

        # TRANSITION BLOCK 3
        self.pool3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2,
                               padding=1, bias=False)  # in = 20x20x32 , out = 10x10x32, RF = 35

        # CONVOLUTION BLOCK 4
        self.convblock4A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 10X10x32 , out = 8X8x32, RF = 26, RF = 39

        self.convblock4B = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )  # in = 8X8x32 , out =6x6x16, RF = 26   , RF = 43

        self.dilated5A = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=2, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )  # in = 6x6x16 , out = 6x6x16, RF = 7        , RF = 51

        # OUTPUT BLOCK
        self.Gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )  # in = 6x6x32 , out = 1x1x32, RF = 54	, RF = 61
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 1x1x32 , out = 1x1x10, RF = 54, RF = 61

    def forward(self, x):
        x = self.pointwise1A(self.depthwise1A(self.convblock1A(x)))
        x = self.pointwise1B(self.depthwise1B(x))
        x = self.pointwise1C(self.depthwise1C(x))
        x = self.pool1(x)
        x = self.pointwise2A(self.depthwise2A(self.convblock2A(x)))
        x = self.pointwise2B(self.depthwise2B(x))
        x = self.pointwise2C(self.depthwise2C(x))
        x = self.pool2(x)
        x = self.pointwise3A(self.depthwise3A(self.convblock3A(x)))
        x = self.pointwise3B(self.depthwise3B(x))
        x = self.pointwise3C(self.depthwise3C(x))
        x = self.pool3(x)
        x = self.convblock4B(self.convblock4A(x))
        x = self.dilated5A(x)
        x = self.fc1(self.Gap1(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class S7_CNNModel_mixed(nn.Module):

    # Mix of depthwise and depthwise-separable convolutions
    def __init__(self):
        super(S7_CNNModel_mixed, self).__init__()

        # CONVOLUTION BLOCK 0
        self.convblock0A = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 32x32x3 , out = 32x32x32, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock1A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 32x32x3 , out = 32x32x64, RF = 3

        self.depthwise1A = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 32x32x64 , out = 32x32x64, RF = 7

        self.depthwise1B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 32x32x64 , out = 32x32x64, RF = 9

        self.depthwise1C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 32x32x64 , out = 32x32x64, RF = 11
        self.pointwise1C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 32x32x1X64 , out = 32x32x32   , RF = 11

        # TRANSITION BLOCK 1
        self.pool1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 32x32x32 , out = 28x28x32  , RF = 15

        # CONVOLUTION BLOCK 2

        self.convblock2A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 28x28x32 , out = 26x26x32, RF = 17

        self.depthwise2A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 26x26x32 * 3x3x32 , out = 26x26x1x32, RF = 19
        self.pointwise2A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 26x26x1X32 , out = 26x26x64    , RF = 19

        self.depthwise2B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 26x26x64 , out = 26x26x64, RF = 21

        self.depthwise2C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 26x26x1x64 , out = 26x26x64, RF = 23
        self.pointwise2C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 26x26x64 , out = 26x26x32  , RF = 23

        # TRANSITION BLOCK 2
        self.pool2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 26x26x32 , out = 22x22x32  , RF = 27

        # CONVOLUTION BLOCK 3
        self.convblock3A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 22x22x32 , out = 20x20x32, RF = 29

        self.depthwise3A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 20x20x32 * 3x3x32 , out = 20x20x1x32, RF = 31
        self.pointwise3A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 20x20x1X32 , out = 20x20x64, RF = 31

        self.depthwise3B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 20x20x64 , out = 20x20x64, RF = 33

        self.depthwise3C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )  # in = 20x20x1x64 , out = 20x20x64, RF = 35
        self.pointwise3C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 20x20x64 , out = 20x20x32      , RF = 35

        # TRANSITION BLOCK 3
        self.pool3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2,
                               padding=1, bias=False)  # in = 20x20x32 , out = 10x10x32, RF = 37

        # CONVOLUTION BLOCK 4
        self.convblock4A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 10X10x32 , out = 8X8x32, RF = 41
        self.pointwise4A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 8x8x32 , out = 8x8x16      , RF = 41

        self.convblock4B = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )  # in = 8X8x16 , out =6x6x16, RF = 45

        self.dilated5A = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=2, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )  # in = 6x6x16 , out = 6x6x16, RF = 53

        # OUTPUT BLOCK
        self.Gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )  # in = 6x6x16 , out = 1x1x16, RF = 63
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 1x1x16 , out = 1x1x10, RF = 63

    def forward(self, x):
        x = self.convblock0A(x)
        x = self.depthwise1A(self.convblock1A(x))
        x = self.depthwise1B(x)
        x = self.pointwise1C(self.depthwise1C(x))
        x = self.pool1(x)
        x = self.pointwise2A(self.depthwise2A(self.convblock2A(x)))
        x = self.depthwise2B(x)
        x = self.pointwise2C(self.depthwise2C(x))
        x = self.pool2(x)
        x = self.pointwise3A(self.depthwise3A(self.convblock3A(x)))
        x = self.depthwise3B(x)
        x = self.pointwise3C(self.depthwise3C(x))
        x = self.pool3(x)
        x = self.convblock4B(self.pointwise4A(self.convblock4A(x)))
        x = self.dilated5A(x)
        x = self.fc1(self.Gap1(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)