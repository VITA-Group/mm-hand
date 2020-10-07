from models.networks import *
from torch import nn

class DepthGenerator(nn.Module):
    def __init__(self):
        super(DepthGenerator, self).__init__()

        models = []
        n_layer = 5
        n_filter = 32
        for i in range(n_layer-1):
            models.append(nn.Sequential(nn.ConvTranspose2d(n_filter, n_filter, kernel_size=6, stride=2, output_padding=0,
                                                           padding=2),
                                        nn.BatchNorm2d(n_filter),
                                        nn.LeakyReLU(0.2, True)))
        # last layer
        models.append(nn.Sequential(nn.ConvTranspose2d(n_filter, 1, kernel_size=6, stride=2, output_padding=0, padding=2),
                                    nn.BatchNorm2d(1),
                                    nn.LeakyReLU(0.2, True),
                                    #nn.Tanh(),
                                    ))

        self.linear = nn.Linear(21 * 2, 4 * 4 * 32)
        self.models = nn.Sequential(*models)

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape((-1, 32, 4, 4))
        return self.models(x)


if __name__ == "__main__":
    import torch
    input = torch.rand(10, 63)
    net = DepthGenerator()
    output = net(input)
    print(output.shape)




