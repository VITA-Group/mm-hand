from models.networks import *


# class Conv2dBlks(nn.Module):
#
#     def __init__(self, settings):
#         super().__init__()
#         self.models = []
#         for setting in settings:
#             self.models.extend(self._make_blk(*setting))
#
#         self.models = nn.Sequential(*self.models)
#
#     def _make_blk(self, in_channels, out_channels, kernel, strides, bias):
#         blk = list()
#         blk.append(nn.Conv2d(in_channels=in_channels,
#                              out_channels=out_channels,
#                              stride=strides,
#                              kernel_size=kernel,
#                              bias=bias))
#         blk.append(nn.ReLU(True))
#         blk.append(nn.BatchNorm2d(out_channels, momentum=0.1))
#         return blk
#
#     def forward(self, input):
#         return self.models(input)
#
#
# class Deconv3dBlks(nn.Module):
#     def __init__(self, settings):
#         super().__init__()
#         self.models = []
#         for setting in settings:
#             self.models.extend(self._make_blk(*setting))
#         self.models = nn.Sequential(*self.models)
#
#     def _make_blk(self, planes, out_planes, kernel, strides, bias):
#         blk = list()
#         blk.append(nn.ConvTranspose3d(planes, out_planes, kernel, strides, bias=bias))
#         blk.append(nn.BatchNorm3d(out_planes, momentum=0.1))
#         blk.append(nn.ReLU(True))
#         return blk
#
#     def forward(self, x):
#         return self.models(x)


# class VoxelGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         settings = [[512, 256, 2, 2, True],
#                     [256, 128, 2, 2, True],
#                     [128,  64, 2, 2, True],
#                     [ 64,   1, 2, 2, True]]
#         self.linear = nn.Linear(21 * 3, 4 * 4 * 4 * 512)
#         self.models = Deconv3dBlks(settings)
#
#     def forward(self, x):
#         input = self.linear(x)
#         batch_size = input.shape[0]
#         input = input.reshape(batch_size, 512, 4, 4, 4)
#         return self.models(input)

# class VoxelGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.encoder = Conv3dBlks([[21, 32, 2, 2, 0,False],
#                                    [32, 16, 2, 2, 0,False],
#                                    [16,  8, 2, 2, 0,False],
#                                    [ 8,  4, 2, 2, 0, False]])
#
#         self.fcs = nn.Sequential(*[Flatten(),
#                     nn.Linear(256, 512),
#                     nn.Linear(512, 1024),
#                     nn.Linear(1024, 2048),])
#
#         self.decoder = Deconv3dBlks([[32, 16, 2, 2,False],
#                                      [16, 8,  2, 2, False],
#                                      [8,  4,  2, 2,  False],
#                                      [4,  1,  2, 2,  False]])
#
#         # self.models = nn.Sequential(*[self.encoder, *self.fcs, self.decoder])
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.fcs(x)
#         x = x.reshape(x.size(0), 32, 4 , 4 ,4)
#         x = self.decoder(x)
#         return x

class VoxelGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(VoxelGenerator, self).__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(1, 16, 7),
            Pool3DBlock(2),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32),
        )

        self.encoder_decoder = EncoderDecoder3D()

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            Basic3DBlock(32, 32, 1),
            Basic3DBlock(32, 32, 1),
            Upsample3DBlock(32, output_channels, 2, 2),
        )

        self.output_layer = nn.Conv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        self.init_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        x = self.tanh(self.output_layer(x))
        return x

    def init_weights(self, *args, **kwargs):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)






