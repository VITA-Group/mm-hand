from torch import nn
from .networks import *

class PoseResNet(nn.Module):

    def __init__(self, block, layers, opt, **kwargs):
        self.inplanes = 64
        extra = opt.POSE_RESNET
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(opt.input_nc, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=opt.num_joints,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding, # 1
                    output_padding=output_padding, # 0
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.deconv_layers(x)
        # print(x.shape)
        x = self.final_layer(x)
        # print(x.shape)
        return x

    def init_weights(self, pretrained=''):

        logger.info('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                logger.info('=> init {}.weight as 1'.format(name))
                logger.info('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        logger.info('=> init final conv weights from normal distribution')
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        # pretrained_state_dict = torch.load(pretrained)
        resnet_models = {50: "resnet50-19c8e357.pth",
                         101: 'resnet101-5d3b4d8f.pth',
                         152: 'resnet152-b121ed2d.pth'}
        checkpoint = torch.hub.load_state_dict_from_url(f'https://s3.amazonaws.com/pytorch/models/{resnet_models[pretrained]}',
                                                        model_dir="./checkpoints")
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict_old = checkpoint['state_dict']
            state_dict = OrderedDict()
            # delete 'module.' because it is saved from DataParallel module
            for key in state_dict_old.keys():
                if key.startswith('module.'):
                    # state_dict[key[7:]] = state_dict[key]
                    # state_dict.pop(key)
                    state_dict[key[7:]] = state_dict_old[key]
                else:
                    state_dict[key] = state_dict_old[key]
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(pretrained))
        self.load_state_dict(state_dict, strict=False)

            # logger.error('=> imagenet pretrained model dose not exist')
            # logger.error('=> please download it first')
            # raise ValueError('imagenet pretrained model does not exist')
