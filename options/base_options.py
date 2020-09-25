import argparse
import os

import torch

from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--imageroot',
                                 type=str,
                                 help='path to images')
        self.parser.add_argument('--poseroot', type=str, help='path to poses')
        self.parser.add_argument('--batchSize',
                                 type=int,
                                 help='input batch size')
        self.parser.add_argument('--fineSize',
                                 type=int,
                                 default=256,
                                 help='then crop to this size')
        '''Added for Depth Map'''
        self.parser.add_argument('--output_nc',
                                 type=int,
                                 default=3,
                                 help='# of output image channels')
        self.parser.add_argument(
            '--ngf',
            type=int,
            default=64,
            help='# of generator filters in first conv layer')
        self.parser.add_argument(
            '--ndf',
            type=int,
            default=64,
            help='# of discrimator filters in first conv layer')
        self.parser.add_argument('--n_layers_D',
                                 type=int,
                                 default=3,
                                 help='blocks used in D')
        self.parser.add_argument(
            '--gpu_ids',
            type=str,
            default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument(
            '--name',
            type=str,
            default='experiment_name',
            help=
            'name of the experiment. It decides where to store samples and models'
        )
        self.parser.add_argument('--nThreads',
                                 default=8,
                                 type=int,
                                 help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir',
                                 type=str,
                                 default='./checkpoints',
                                 help='models are saved here')
        self.parser.add_argument(
            '--norm',
            type=str,
            default='batch',
            help='instance normalization or batch normalization')
        self.parser.add_argument(
            '--serial_batches',
            action='store_true',
            help=
            'if true, takes images in order to make batches, otherwise takes them randomly'
        )
        self.parser.add_argument('--display_winsize',
                                 type=int,
                                 default=256,
                                 help='display window size')
        self.parser.add_argument('--display_id',
                                 type=int,
                                 default=0,
                                 help='window id of the web display')
        self.parser.add_argument('--display_port',
                                 type=int,
                                 default=8097,
                                 help='visdom port of the web display')
        self.parser.add_argument('--no_dropout',
                                 action='store_true',
                                 help='no dropout for the generator')
        self.parser.add_argument(
            '--max_dataset_size',
            type=int,
            default=float("inf"),
            help=
            'Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.'
        )
        self.parser.add_argument(
            '--no_flip',
            action='store_true',
            help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument(
            '--init_type',
            type=str,
            default='normal',
            help='network initialization [normal|xavier|kaiming|orthogonal]')

        self.parser.add_argument('--H_input_nc',
                                 type=int,
                                 default=3,
                                 help='# of input image channels')
        '''Added'''
        self.parser.add_argument('--P_input_nc',
                                 type=int,
                                 default=21,
                                 help='# of input image channels')
        self.parser.add_argument('--D_input_nc',
                                 type=int,
                                 default=3,
                                 help='# of input image channels')
        self.parser.add_argument('--padding_type',
                                 type=str,
                                 default='reflect',
                                 help='# of input image channels')
        self.parser.add_argument('--pairLst', type=str, help='market pairs')

        self.parser.add_argument('--use_flip',
                                 type=int,
                                 default=0,
                                 help='flip or not')

        # down-sampling times
        self.parser.add_argument('--G_n_downsampling',
                                 type=int,
                                 default=2,
                                 help='down-sampling blocks for generator')
        self.parser.add_argument('--D_n_downsampling',
                                 type=int,
                                 default=2,
                                 help='down-sampling blocks for discriminator')

        # special params
        self.parser.add_argument("--augmentation_ratio", type=float)
        self.parser.add_argument("--augmentation_method", type=str)
        self.parser.add_argument("--dataset_mode", type=str)
        self.parser.add_argument("--dataset", type=str)
        self.parser.add_argument("--dataroot", type=str)
        #distributed options
        self.parser.add_argument("--local_rank",
                                 default=0,
                                 type=int,
                                 help="determine which is the master process")
        self.parser.add_argument(
            "--distributed",
            action='store_true',
            help="let program know you are running in distributed mode")
        self.parser.add_argument(
            "--seed",
            type=int,
            default=49,
            help="set manual seed for uniform weight init for DDP process")
        self.parser.add_argument("--opt_level", default='O0', type=str)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        opt.isTrain = self.isTrain  # train or test

        if opt.distributed:
            opt.gpu = opt.local_rank
            torch.cuda.set_device(opt.gpu)
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            opt.world_size = torch.distributed.get_world_size()
            opt.lr = opt.lr
            opt.batchSize = opt.batchSize // opt.world_size
        else:
            str_ids = opt.gpu_ids.split(',')
            opt.gpu_ids = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    opt.gpu_ids.append(id)
            if len(opt.gpu_ids) > 0:
                torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        torch.backends.cudnn.benchmark = True
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
