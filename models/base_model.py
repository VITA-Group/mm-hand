import os

import torch
import torch.nn as nn
from apex import amp


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.gpu_ids = [opt.local_rank]
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.master = opt.local_rank == 0

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        if self.master:
            save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
            save_path = os.path.join(self.save_dir, save_filename)

            if self.opt.distributed:
                torch.save(network.module.state_dict(), save_path)
                save_path = save_path.replace(network_label, 'amp')
                torch.save(amp.state_dict(), save_path)
            else:
                torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self):
        opt = self.opt
        weights_paths = [
            os.path.join('checkpoints', opt.name, i)
            for i in os.listdir(os.path.join("checkpoints", opt.name))
            if opt.which_epoch in i
        ]
        for path in weights_paths:
            if 'amp' in path:
                try:
                    amp.load_state_dict(torch.load(path))
                except:
                    pass
            else:
                weights = torch.load(path)
                name = path.split("/")[-1]  # get the name of weights
                name = name[0:-4]  # get rid of .pth
                name = name.replace(f"{opt.which_epoch}_net_", '')
                sub_model = getattr(self, name)
                sub_model.load_state_dict(weights)
                print(f"loading weights for {name}")

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
