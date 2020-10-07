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
import numpy as np
import apex
from apex import amp

# Hand Pose Module 2-D
class hpm3dModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument("--data_mode", type=str, default='hpm3d')
        parser.add_argument("--num_loss", type=int, default=1)
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
        self.loss_names = ['L_z', 'null']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = []
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['Hpm3d']

        self.loss_null = torch.tensor(0).to(self.device)
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.optimizer = torch.optim.Adam if not self.opt.distributed else apex.optimizers.FusedAdam

        self.networkFactory = networks.NetworkInitializer(networks=['hpm3d'],
                                                          optimizers={self.optimizer: 'hpm3d'},
                                                          options=opt)

        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionLoss = torch.nn.SmoothL1Loss()
            [self.netHpm3d], [self.optimizer] = self.networkFactory()

            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizers = [self.optimizer]
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks
        else:
            [self.netHpm3d], _ = self.networkFactory()


    def set_input(self, input):
        """Unpack input dataset from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the dataset itself and its metadata information.
        """

        self.heatmaps = input['A'].to(self.device) # there are 21
        self.ground_truth = torch.squeeze(input['B'].to(self.device), dim=-1)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.output = self.netHpm3d(self.heatmaps)  # generate prediction and loss

    def backward(self):
        """
        Calculate losses, gradients, and update network weights.
        In DGGAN, this loss if L_2D using simple MSE loss
        Authors  amply this loss by 100 and not mention it.

        """
        self.loss_L_z = self.criterionLoss(self.output, self.ground_truth) * 10
        self.loss_backward(self.loss_L_z, self.optimizer)

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
