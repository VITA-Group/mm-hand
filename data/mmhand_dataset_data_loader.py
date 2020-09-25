import numpy as np
import torch.utils.data

from data.mmhand_dataset import MMHandDataset
from data.rhd_dataset import RHDdataset
from data.stb_dataset import STBdataset


class MMHandDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        if opt.dataset == 'stb':
            self.dataset = STBdataset(opt)
        elif opt.dataset == 'rhd':
            self.dataset = RHDdataset(opt)
        else:
            self.dataset = MMHandDataset(opt)

        def _init_fn(w_id):
            np.random.seed(opt.seed)
        self.distributed_sampler = None
        if self.opt.distributed:
            self.distributed_sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset)
            self.func = _init_fn
            if self.opt.local_rank == 0:
                print("dataset [%s] was created" % type(self.dataset).__name__)
        else:
            print("dataset [%s] was created" % type(self.dataset).__name__)
            self.func = None

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=False,
            pin_memory=True,
            sampler=self.distributed_sampler,
            worker_init_fn=self.func,
            num_workers=int(opt.nThreads))

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
