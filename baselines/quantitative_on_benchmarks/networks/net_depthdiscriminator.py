from models.networks import *
from torch import nn


class DepthDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, output_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.encoder = [
            nn.Conv2d(input_nc + output_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]

        self.classifier = nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)

        self.predictor = [nn.Conv2d(ndf*2, 21, kernel_size=1, stride=1, padding=0, bias=use_bias),
                          nn.LeakyReLU(0.2, True),
                          norm_layer(21),
                          nn.LeakyReLU(0.2, True),]

        self.hard_predictor = [nn.Conv2d(ndf*2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
                               nn.LeakyReLU(0.2, True),
                               norm_layer(1),
                               nn.LeakyReLU(0.2, True),
                               Flatten(),
                               nn.Linear(1*256*256, 256),
                               nn.Linear(256, 63)]

        self.encoder = nn.Sequential(*self.encoder)
        self.predictor = nn.Sequential(*self.predictor)
        self.hard_predictor = nn.Sequential(*self.hard_predictor)

    def forward(self, input):
        """Standard forward."""
        input = self.encoder(input)
        classification = self.classifier(input)
        heatmaps = self.predictor(input)
        keypoints = self.hard_predictor(input)

        return classification, heatmaps, keypoints

if __name__ == "__main__":
    input = torch.rand(90, 1, 128, 128)
    net = DepthDiscriminator(1)
    output1, output2, output3 = net(input)
    print(output1.shape, output2.shape, output3.shape)

