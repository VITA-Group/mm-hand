import itertools
import os
from collections import OrderedDict

import apex
import numpy as np
import torch
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

import util.util as util
from .base_model import BaseModel
from .Discriminator import Discriminator
from .Generator import Generator
from .network_utils import GANLoss
from .network_utils import get_norm_layer
from .network_utils import get_scheduler
from .network_utils import init_weights
from .network_utils import print_network
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss
from util.image_pool import ImagePool
# losses


class MMHandModel(BaseModel):
    def name(self):
        return 'MMHandModel'

    def __init__(self, opt):
        super(MMHandModel, self).__init__(opt)
        '''Added for Depth Map'''
        nb = opt.batchSize
        size = opt.fineSize
        self.overflow = False
        input_nc = [
            opt.H_input_nc, opt.P_input_nc + opt.P_input_nc,
            opt.D_input_nc + opt.D_input_nc
        ]
        self.netG = self.define_G(input_nc,
                                  opt.output_nc,
                                  opt.ngf,
                                  opt.norm,
                                  not opt.no_dropout,
                                  opt.init_type,
                                  self.gpu_ids,
                                  n_downsampling=opt.G_n_downsampling)

        if self.isTrain:
            self.netD_PB = self.define_D(opt.H_input_nc + opt.P_input_nc,
                                         opt.ndf,
                                         opt.n_layers_D,
                                         opt.norm,
                                         opt.no_lsgan,
                                         opt.init_type,
                                         self.gpu_ids,
                                         not opt.no_dropout_D,
                                         n_downsampling=opt.D_n_downsampling)

            self.netD_PP = self.define_D(opt.H_input_nc + opt.H_input_nc,
                                         opt.ndf,
                                         opt.n_layers_D,
                                         opt.norm,
                                         opt.no_lsgan,
                                         opt.init_type,
                                         self.gpu_ids,
                                         not opt.no_dropout_D,
                                         n_downsampling=opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train:
            self.load_network()

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan,
                                        gpu=self.opt.local_rank)

            if opt.L1_type == 'origin':
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'l1_plus_perL1':
                self.criterionL1 = L1_plus_perceptualLoss(
                    opt.lambda_A, opt.lambda_B, opt.perceptual_layers,
                    self.gpu_ids, opt.percep_is_l1)
            else:
                raise Exception('Unsurportted type of L1!')
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(),
                                                   lr=opt.lr,
                                                   betas=(opt.beta1, 0.999))
            self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(),
                                                   lr=opt.lr,
                                                   betas=(opt.beta1, 0.999))
            if self.opt.distributed:
                [self.netG, self.netD_PB, self.netD_PP], \
                        [self.optimizer_G, self.optimizer_D_PB, self.optimizer_D_PP] = \
                        amp.initialize(models=[self.netG,
                                               self.netD_PB,
                                               self.netD_PP],
                                       optimizers=[self.optimizer_G,
                                                   self.optimizer_D_PB,
                                                   self.optimizer_D_PP],
                                       num_losses=3, opt_level=opt.opt_level)
                self.netG = apex.parallel.convert_syncbn_model(
                    DDP(self.netG, delay_allreduce=True))

                self.netD_PB = apex.parallel.convert_syncbn_model(
                    DDP(self.netD_PB, delay_allreduce=True))

                self.netD_PP = apex.parallel.convert_syncbn_model(
                    DDP(self.netD_PP, delay_allreduce=True))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_PB)
            self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(get_scheduler(optimizer, opt))

        if self.master:
            print('---------- Networks initialized -------------')
            print_network(self.netG)
            if self.isTrain:
                print_network(self.netD_PB)
                print_network(self.netD_PP)
                print(opt.local_rank)
            print('-----------------------------------------------')

    def define_G(self,
                 input_nc,
                 output_nc,
                 ngf,
                 norm='batch',
                 use_dropout=False,
                 init_type='normal',
                 gpu_ids=[],
                 n_downsampling=2):
        use_gpu = len(gpu_ids) > 0
        norm_layer = get_norm_layer(norm_type=norm)

        if use_gpu:
            assert (torch.cuda.is_available())

        assert len(input_nc) == 3
        netG = Generator(input_nc,
                         output_nc,
                         ngf,
                         norm_layer=norm_layer,
                         use_dropout=use_dropout,
                         n_blocks=9,
                         gpu_ids=gpu_ids,
                         n_downsampling=n_downsampling)
        if not self.opt.distributed:
            if len(gpu_ids) > 0:
                netG.cuda(gpu_ids[0])
        else:
            netG = netG.to(self.opt.gpu)
        init_weights(netG, init_type=init_type)
        return netG

    def define_D(self,
                 input_nc,
                 ndf,
                 n_layers_D=3,
                 norm='batch',
                 use_sigmoid=False,
                 init_type='normal',
                 gpu_ids=[],
                 use_dropout=False,
                 n_downsampling=2):
        use_gpu = len(gpu_ids) > 0
        norm_layer = get_norm_layer(norm_type=norm)

        if use_gpu:
            assert (torch.cuda.is_available())

        netD = Discriminator(input_nc,
                             ndf,
                             norm_layer=norm_layer,
                             use_dropout=use_dropout,
                             n_blocks=n_layers_D,
                             gpu_ids=[],
                             padding_type='reflect',
                             use_sigmoid=False,
                             n_downsampling=n_downsampling)
        if not self.opt.distributed:
            if use_gpu:
                netD.cuda(gpu_ids[0])
        else:
            netD = netD.to(self.opt.gpu)
        init_weights(netD, init_type=init_type)
        return netD

    def set_input(self, input):
        input_H1, input_P1, input_D1 = input['H1'], input['P1'], input['D1']
        input_H2, input_P2, input_D2 = input['H2'], input['P2'], input['D2']
        '''Added for Depth Map'''

        self.input_H1 = input_H1.to(self.gpu_ids[0])
        self.input_P1 = input_P1.to(self.gpu_ids[0])
        self.input_D1 = input_D1.to(self.gpu_ids[0])
        '''Added for Depth Map'''
        self.input_H2 = input_H2.to(self.gpu_ids[0])
        self.input_P2 = input_P2.to(self.gpu_ids[0])
        self.input_D2 = input_D2.to(self.gpu_ids[0])

        self.image_paths = input['H1_path'][0] + '___' + input['H2_path'][0]

    def forward(self):
        G_input = [
            self.input_H1,
            torch.cat((self.input_P1, self.input_P2), 1),
            torch.cat((self.input_D1, self.input_D2), 1)
        ]
        self.fake_p2 = self.netG(G_input)

    def test(self):
        '''Added for Depth Map'''
        G_input = [
            self.input_H1,
            torch.cat((self.input_P1, self.input_P2), 1),
            torch.cat((self.input_D1, self.input_D2), 1)
        ]
        self.fake_p2 = self.netG(G_input)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):

        pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_P2),
                                              1))
        self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_H1),
                                              1))
        self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)

        # L1 loss
        losses = self.criterionL1(self.fake_p2, self.input_H2)
        self.loss_G_L1 = losses[0]
        self.loss_originL1 = losses[1].data
        self.loss_perceptual = losses[2].data

        pair_L1loss = self.loss_G_L1
        pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
        pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
        pair_GANloss /= 2

        pair_loss = pair_L1loss + pair_GANloss

        self.loss_backward(pair_loss, self.optimizer_G, 0)
        self.pair_L1loss = pair_L1loss.data
        self.pair_GANloss = pair_GANloss.data

    def backward_D_basic(self, netD, real, fake, optimizer, iid):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        self.loss_backward(loss_D, optimizer, iid)
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self):
        real_PB = torch.cat((self.input_H2, self.input_P2), 1)
        fake_PB = self.fake_PB_pool.query(
            torch.cat((self.fake_p2, self.input_P2), 1).data)
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB,
                                          self.optimizer_D_PB, 1)
        self.loss_D_PB = loss_D_PB.data

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_H2, self.input_H1), 1)
        fake_PP = self.fake_PP_pool.query(
            torch.cat((self.fake_p2, self.input_H1), 1).data)
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP,
                                          self.optimizer_D_PP, 2)
        self.loss_D_PP = loss_D_PP.data

    def loss_backward(self, loss, optimizers, id=0):
        if self.opt.distributed:
            default_optimizer_step = optimizers.step
            with amp.scale_loss(loss, optimizers, id) as scaled_loss:
                scaled_loss.backward()
            # check if optimizer overflow
            self._overflow = torch.tensor(0).to(self.opt.local_rank)
            if not optimizers.step is default_optimizer_step:
                self._overflow = torch.tensor(1).to(self.opt.local_rank)

            if dist.get_world_size() > 1:
                self.overflow = reduce_tensor(
                    self._overflow, dist.get_world_size()) > 0 or self.overflow
        else:
            loss.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        if not self.overflow: self.optimizer_G.step()

        # D_P
        for i in range(self.opt.DG_ratio):
            self.optimizer_D_PP.zero_grad()
            self.backward_D_PP()
            if not self.overflow: self.optimizer_D_PP.step()

        # D_BP
        for i in range(self.opt.DG_ratio):
            self.optimizer_D_PB.zero_grad()
            self.backward_D_PB()
            if not self.overflow: self.optimizer_D_PB.step()

        self.overflow = False

    def get_current_errors(self):
        ret_errors = OrderedDict([('pair_L1loss', self.pair_L1loss)])
        ret_errors['D_PP'] = self.loss_D_PP
        ret_errors['D_PB'] = self.loss_D_PB
        ret_errors['pair_GANloss'] = self.pair_GANloss

        ret_errors['origin_L1'] = self.loss_originL1
        ret_errors['perceptual'] = self.loss_perceptual

        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_H1.size(2), self.input_H1.size(3)
        input_H1 = util.tensor2im(self.input_H1.data)
        input_H2 = util.tensor2im(self.input_H2.data)

        input_P1 = util.draw_pose_from_map(self.input_P1.data)
        input_P2 = util.draw_pose_from_map(self.input_P2.data)
        '''Added for Depth Map'''
        input_D1 = util.tensor2im(self.input_D1.data)
        input_D2 = util.tensor2im(self.input_D2.data)

        fake_p2 = util.tensor2im(self.fake_p2.data)
        '''Added for Depth Map'''
        vis = np.zeros((height, width * 7, 3)).astype(np.uint8)  #h, w, c
        vis[:, :width, :] = input_H1
        vis[:, width:width * 2, :] = input_P1
        vis[:, width * 2:width * 3, :] = input_D1

        vis[:, width * 3:width * 4, :] = input_H2
        vis[:, width * 4:width * 5, :] = input_P2
        vis[:, width * 5:width * 6, :] = input_D2

        vis[:, width * 6:, :] = fake_p2

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG, 'netG', label, self.gpu_ids)
        self.save_network(self.netD_PB, 'netD_PB', label, self.gpu_ids)
        self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)

    def pprint(self, msg):
        if self.master:
            print(msg)


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt
