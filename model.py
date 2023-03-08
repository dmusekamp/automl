import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        """  Implements a residual block with convolutional layers.
        Uses a convolutional layer in the skip connection if applying downsampling.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size for all convolutional layers. Has to be an odd number.
        :param stride: stride of the first convolutional layer
        """
        super().__init__()
        padding = 'same' if stride == 1 else int(kernel_size / 2)
        assert kernel_size % 2 == 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.skip_conv = None
        if stride != 1 or in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, stride)

    def forward(self, x):
        """ Runs the residual block on a 2d input."""
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))

        if self.skip_conv is not None:
            out += self.skip_conv(x)
        else:
            out += x

        return F.relu(out)
