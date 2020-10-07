import torch
import torch.nn as nn
from torchvision.models import vgg19 as pretrainVgg19


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        #STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
        # self.style_list = [2, 9, 16, 29, 42]
        self.style_list = [1, 6, 11, 20, 29]

        self.content_list = [22]
        features = list(pretrainVgg19(pretrained=True).features)[0:30]
        self.features = nn.ModuleList(features).eval()
        # self.norm = Normalization(torch.tensor([0.485, 0.456, 0.406]),
        #                           torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, x, mode='style'):
        features = self.style_list if mode == 'style' else self.content_list
        results = []
        # x = self.norm(x)
        for i, model in enumerate(self.features):
            x = model(x)
            if i in features:
                results.append(x)
        return results

    def init_weights(self, *args, **kwargs):
        pass

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        if img.device != self.mean.device:
            self.mean = self.mean.to(img.device)
            self.std = self.std.to(img.device)
        return (img - self.mean) / self.std



if __name__ == "__main__":
    net = Vgg19()