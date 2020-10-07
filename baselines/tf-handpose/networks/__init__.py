# from .networks import *
import torch
import importlib
import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from .blocks import *
from torch.optim import lr_scheduler
from torch.nn import init
import sys
from typing import List, Callable, Dict

from easydict import EasyDict as edict
import itertools
import inspect


class NetworkInitializer:
    def __init__(self,
                 networks  : Dict[str, Callable] or List[str],
                 optimizers,
                 options):
        """

        :param networks: name of network file and optional custom initializer. Note custom init should take in the network as well as the options
        :param optimizers: dictionary of optimizers and list of networks for that optimizers
        :param options: general options from BaseOptions
        """
        super().__init__()
        self.opt = options
        self.networks_options = networks
        self.optimizers_options = optimizers

        self.nets = None
        self.optimizers = None

        self.updated = False

    def __call__(self, *args, **kwargs):
        return self.get_networks()

    def __repr__(self):
        pass

    def add_net(self, networks: Dict[str, Callable] or str):
        pass

    def add_optimizers(self, optimizers: Dict[Callable, List[str] or str]):
        pass

    def _initialize_network(self):
        self.nets = {}
        input_t = self.networks_options.__class__
        for k in self.networks_options:
            net = find_network_using_name(k)
            initializer = define_network if input_t == list else self.networks_options[k]
            self.nets[k] = initializer(net, self.opt)
            init_weights(self.nets[k], self.opt)

    def _initialize_optimizers(self):
        self.optimizers = []

        if self.optimizers_options is None:
            Warning("no optimizers options given for training")
        else:
            for k, v in self.optimizers_options.items():
                #print(k, v)
                params = itertools.chain(*[self.nets[i].parameters() for i in v]) \
                    if v.__class__ is list else self.nets[v].parameters()
                # currently only support Adam optimizers
                k = k(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
                self.optimizers.append(k)

    def _networks_DDP(self):
        """ network distributed data parallel"""
        self._initialize_network()

        device = self.opt.local_rank
        isTrain = self.opt.isTrain

        for name, net in self.nets.items():
            self.nets[name] = net.to(device)

        netlist = [v for _, v in self.nets.items()]
        if isTrain:
            self._initialize_optimizers()
            netlist, opts = amp.initialize(models=netlist,
                                           optimizers=self.optimizers,
                                           opt_level=self.opt.opt_level,
                                           num_losses=self.opt.num_loss)
        else:
            netlist = amp.initialize(models=netlist, opt_level=self.opt.opt_level)

        for i, net in enumerate(netlist):
            net = DDP(net, delay_allreduce=True)
            netlist[i] = apex.parallel.convert_syncbn_model(net)

        self.nets = {k: v for k, v in zip(self.nets, netlist)}
        if not isTrain:
            for k in self.nets:
                self.nets[k].eval()

    def _networks_DP(self):
        """network data parrallel"""
        self._initialize_network()

        if self.opt.isTrain:
            self._initialize_optimizers()

        if len(self.opt.gpu_ids) > 0:
            for k, v in self.nets.items():
                v = v.to(self.opt.gpu_ids[0])
                self.nets[k] = torch.nn.DataParallel(v, self.opt.gpu_ids)

        if not self.opt.isTrain:
            for k in self.nets:
                self.nets[k].eval()

    def get_networks(self):
        if self.nets is None or self.updated:
            if self.opt.distributed:
                self._networks_DDP()
            else:
                self._networks_DP()
        return [v for _, v in self.nets.items()], self.optimizers



def define_network(net, opt: edict or Dict):
    """ boiler plate network initialization"""
    specs = inspect.getfullargspec(net)
    args, defaults = specs.args[1::], specs.defaults if specs.defaults is not None else []

    class NoDefaultParam:
        def __init__(self):
            pass

    defaults = [*[NoDefaultParam() for _ in range(abs(len(args) - len(defaults)))],
                *defaults]

    params = {k:v for k, v in zip(args, defaults)}

    #overwrite params with options params
    for k, v in params.items():
        params[k] = getattr(opt, k, v)
        if params[k].__class__ == NoDefaultParam:
            print(f"Required Parameter {k} not found in Options for network {net.__name__}"
                  f"Consider supplementing custom network init functions or adding parameter in Option ")
            exit(0)
    return net(**params)


def find_network_using_name(net_name):
    """template from models.__init__.find_model_using_name(..)
    """
    net_filename = f"{sys.modules[__name__].__name__}.net_{net_name}"
    netlib = importlib.import_module(net_filename)
    net = None
    target_net_name = net_name
    for name, cls in netlib.__dict__.items():
        if name.lower() == target_net_name.lower() \
                and issubclass(cls, torch.nn.Module):
            net = cls

    if net is None:
        print(
            "In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (
                net_filename, target_net_name))
        exit(0)

    return net


def find_functions(func_name):
    """find the define functions within this module"""
    func_name = f"define_{func_name}"
    this_module = sys.modules[__name__]
    for name, funcs in this_module.__dict__.items():
        if name.lower() == func_name.lower():
            return funcs
    raise ValueError(f"no function name {func_name}")
    # return getattr(this_module, func_name)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights2(net, init_type='normal', init_gain=0.02, opt=None):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)

    if hasattr(net, f"init_weights"):
        net.init_weights(opt.resnet_size)
    else:
        net.apply(init_func)  # apply the initialization function <init_func>

def init_weights(net, opt):
    return init_weights2(net, opt.init_type, opt.init_gain)


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_networks(funcs, options, optimizers=None):
    '''
    a clean api to initialize multiple networks, optimizers alongside with/without auto mixed precision
    :param funcs: a list of string with the name of the define functions
    :param optimizers: a list of torch.nn.optim functions
    :return: a list of networks, a list of optimizers init with amp
    '''
    networks = []
    for func_name in funcs:
        net = find_functions(func_name)(options)
        networks.append(net)

    opts = []
    if options.isTrain:
        for i, opt in enumerate(optimizers):
            opts.append(opt(networks[i].parameters(), lr=options.lr, betas=(options.beta1, 0.999)))

    if options.distributed:
        device = options.gpu  # if options.distributed else options.gpu_ids[0]
        for net in networks:
            # init_weights(net, options)
            net.to(device)

        if options.isTrain:
            networks, opts = amp.initialize(models=networks, optimizers=opts, opt_level=options.opt_level,
                                            num_losses=options.num_loss)
        else:
            networks = amp.initialize(models=networks, opt_level=options.opt_level)

        for i, net in enumerate(networks):
            net = DDP(net, delay_allreduce=True)
            networks[i] = apex.parallel.convert_syncbn_model(net)

    if len(networks) == 1:
        networks = networks[0]
    if len(opts) == 1:
        opts = opts[0]

    return networks, opts

def init_net2(net, init_type='normal', init_gain=0.02, gpu_ids=[], opt=None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    print(f"gpu_ids: {gpu_ids}")
    if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights2(net, init_type, init_gain=init_gain, opt=opt)

    # if not opt.distributed and not opt.use_mixed_precision:
    #     if len(gpu_ids) > 0:
    #         assert(torch.cuda.is_available())
    #         net.to(gpu_ids[0])
    #         net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # init_weights(net, init_type, init_gain=init_gain, opt=opt)
    return net

def init_net(net, options):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if not options.distributed:
        if len(options.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            net.to(options.gpu_ids[0])
            net = torch.nn.DataParallel(net, options.gpu_ids)  # multi-GPUs

    init_weights(net, options)

    return net


def define_G(options):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    if options.netG.split('_')[0] == 'resnet':
        net = find_network_using_name('resnetgenerator')
    else:
        net = find_network_using_name('unetgenerator')

    norm_layer = get_norm_layer(norm_type=options.norm)
    use_dropout = True if not options.no_dropout else False

    if options.netG == 'resnet_9blocks':
        net = net(options.input_nc, options.output_nc, options.ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                  n_blocks=9)
    elif options.netG == 'resnet_6blocks':
        net = net(options.input_nc, options.output_nc, options.ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                  n_blocks=6)
    elif options.netG == 'unet_128':
        net = net(options.input_nc, options.output_nc, 7, options.ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif options.netG == 'unet_256':
        net = net(options.input_nc, options.output_nc, 8, options.ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % options.netG)
    return init_net(net, options)


def define_D(options):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """

    net = None
    input_nc = options.input_nc + options.output_nc

    if options.netD == 'pixel':
        net = find_network_using_name('pixeldiscriminator')
    else:
        net = find_network_using_name('nlayerdiscriminator')

    norm_layer = get_norm_layer(norm_type=options.norm)

    if options.netD == 'basic':  # default PatchGAN classifier
        net = net(input_nc, options.ndf, n_layers=3, norm_layer=norm_layer)
    elif options.netD == 'n_layers':  # more options
        net = net(input_nc, options.ndf, options.n_layers_D, norm_layer=norm_layer)
    elif options.netD == 'pixel':  # classify if each pixel is real or fake
        net = net(input_nc, options.ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % options.netD)
    return init_net(net, options)


def define_G1(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    if netG.split('_')[0] == 'resnet':
        net = find_network_using_name('resnetgenerator')
    else:
        net = find_network_using_name('unetgenerator')
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = net(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = net(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = net(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = net(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net2(net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


def define_D1(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None

    if netD == 'pixel':
        net = find_network_using_name('pixeldiscriminator')
    else:
        net = find_network_using_name('nlayerdiscriminator')
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = net(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = net(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = net(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net2(net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)

def define_voxelG(options):
    net = find_network_using_name('voxelgenerator')()
    return init_net(net, options)


def define_voxelD(options):
    net = find_network_using_name('voxeldiscriminator')()
    return init_net(net, options)


def define_depthG(opt):
    net = find_network_using_name('depthgenerator')()
    return init_net(net, opt)


def define_depthD(opt):
    net = find_network_using_name('depthdiscriminator')
    input_nc = opt.input_nc + opt.output_nc
    norm_layer = get_norm_layer(norm_type=opt.norm)
    net = net(input_nc, opt.ndf, norm_layer=norm_layer)

    return init_net(net, opt)


def define_vgg19(opt):
    net = find_network_using_name('vgg19')()
    return init_net(net, opt)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def define_poseNet(opt):
    num_layers = opt.resnet_size
    # style = opt.posenet_style
    block_class, layers = resnet_spec[num_layers]

    # if style == 'caffe':
    #     block_class = Bottleneck_CAFFE

    model = find_network_using_name('poseresnet')(block_class, layers, opt)

    return init_net(model, options=opt)
