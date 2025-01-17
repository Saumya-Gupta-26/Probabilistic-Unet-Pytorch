import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import init_weights
import pdb

class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, initializers, padding, pool=True):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, initializers, padding, pool=False)

    def padfix(self, x2, x1): # NCDH args # added by saumya
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        #pdb.set_trace()
        if diffZ > 0:
            x1 = torch.nn.functional.pad(x1, [0,0, diffZ // 2, diffZ - diffZ // 2])
        elif diffZ < 0:
            x2 = torch.nn.functional.pad(x2, [0,0, diffZ // 2, diffZ - diffZ // 2])

        if diffY > 0:
            x1 = torch.nn.functional.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        elif diffY < 0:
            x2 = torch.nn.functional.pad(x2, [diffY // 2, diffY - diffY // 2, 0, 0])

        return x2, x1

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)

        up, bridge = self.padfix(up, bridge)

        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out =  self.conv_block(out)

        return out