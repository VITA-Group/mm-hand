import torch
import torch.nn as nn
from models.networks import *

class LeakyConv3dBlocks(nn.Module):

    def __init__(self, settings):
        super().__init__()
        self.models = []
        for setting in settings:
            self.models.extend(self._make_blk(*setting))

        self.models = nn.Sequential(*self.models)

    def _make_blk(self, in_channels, out_channels, kernel, strides, padding, bias):
        blk = list()
        blk.append(nn.Conv3d(in_channels=in_channels,
                             out_channels=out_channels,
                             stride=strides,
                             kernel_size=kernel,
                             padding=padding,
                             bias=bias))
        blk.append(nn.BatchNorm3d(out_channels, momentum=0.1))
        blk.append(nn.LeakyReLU(0.2, True))
        return blk

    def forward(self, input):
        return self.models(input)


class VoxelDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.models = LeakyConv3dBlocks([[2, 64, 3, 2, 1, False],
                                         [64, 128,  3, 2, 1, False],
                                         [128, 256, 3, 2, 1, False],
                                         [256, 1, 4, 3, 1, False]])
        # self.models = nn.Sequential(
        #     Basic3DBlock(2, 64, 3),
        #     Pool3DBlock(2),
        #     Basic3DBlock(64, 128, 3),
        #     Pool3DBlock(2),
        #     Basic3DBlock(128, 256, 3),
        #     Pool3DBlock(2),
        #     Basic3DBlock(256, 1, 3),
        # )


    def forward(self, input):
        return self.models(input)