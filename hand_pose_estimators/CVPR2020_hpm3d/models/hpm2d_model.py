"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input dataset and perform dataset pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks
import apex
from apex import amp

# Hand Pose Module 2-D
class hpm2dModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument("--data_mode", type=str, default='hpm2d')
        parser.add_argument('--num_loss', type=int, default=1)
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        # Hpm2d.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        super().__init__(opt)
        self.loss_names = ['mse', 'null']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['ground_truth', 'l6', 'data_A']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['Hpm']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.optimizer = torch.optim.Adam if not self.opt.distributed else apex.optimizers.FusedAdam
        self.loss_null = torch.tensor(0).to(self.device)

        self.networkFactory = networks.NetworkInitializer(networks=['hpm2d'],
                                                          optimizers={self.optimizer: 'hpm2d'},
                                                          options=opt)

        # self.netHpm.cuda()
        if self.isTrain:  # only defined during training time
            self.criterionLoss = Criterion().to(self.device)
            self.criterionL1 = torch.nn.SmoothL1Loss(reduction='mean').to(self.device)

            [self.netHpm], [self.optimizer] = self.networkFactory()

            self.optimizers = [self.optimizer]
        else:
            [self.netHpm], _ = self.networkFactory()
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input dataset from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the dataset itself and its metadata information.
        """
        # data_A is input['B'] because we exepect to see the heatmap
        self.data_A = input['A'].to(self.device)
        self.heatmaps = input['B'].to(self.device)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""

        self.output = self.netHpm(self.data_A)  # generate prediction and loss
        # self.output = results['pred']
        # self.loss_mse = results['loss']
        # print(self.output.shape, self.loss_mse.shape)

    def backward(self):

        self.loss_mse = self.criterionLoss(self.output, self.heatmaps)
        #self.loss_rmse = torch.sqrt(self.loss_mse)
        #self.loss_l1 = self.localization_loss()
        self.loss_backward(self.loss_mse, self.optimizer)


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def compute_visuals(self):
        self.l6 = torch.unsqueeze(torch.unsqueeze(torch.sum(self.output[5][0], dim=0), 0), 0)
        self.ground_truth = torch.unsqueeze(torch.unsqueeze(torch.sum(self.heatmaps[0], dim=0), 0), 0)

    def localization_loss(self):
        ratio = 1.0 / len(self.output)
        loss = 0.0
        BATCH, N_KP, H, W = self.heatmaps.shape

        gt = self.heatmaps.view(BATCH, N_KP, -1)
        gt = gt.permute(1, 0, 2)
        gt_z, gt_i = torch.max(gt, 2)

        d = torch.zeros(gt_i.size()).to(self.device)

        for layer in self.output:
            layer = layer.view(BATCH, N_KP, -1)
            layer = layer.permute(1, 0, 2)
            layer_z, layer_i = torch.max(layer, 2)
            loss += (self.criterionL1(layer_z, gt_z) + self.criterionL1(gt_i.float()-gt_z.float(), d)) * ratio

        return loss * 0.001








class Criterion(torch.nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(reduction)

    def forward(self, input, target):
        loss = 0
        for i in input:
            loss += super().forward(i, target)
        return loss * 1000

