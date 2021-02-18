import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
        A simple ResNet block (skip connections).
        input ___
          |     |
        conv    |
          |     |
        relu    |
          |     |
        conv    |
          |     /
        residual
          |
        relu
          |
        output
    """
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        # out = F.relu(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = F.relu(out)
        return out


class ImpalaBlock(nn.Module):
    """
        A higher level block of convs.
        input
          |
        conv
          |
        relu
          |
        maxpool
          |
        resblock
          |
        resblock
          |
        output
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res_block1 = ResBlock(out_channels)
        self.res_block2 = ResBlock(out_channels)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x