"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import numpy as np


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset)
        self.dataset = dataset_class(opt)
        self.distributed_sampler = None

        def _init_fn(w_id):
            np.random.seed(opt.seed)

        if self.opt.distributed:
            self.distributed_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
            self.func = _init_fn
            if self.opt.local_rank == 0:
                print("dataset [%s] was created" % type(self.dataset).__name__)
        else:
            print("dataset [%s] was created" % type(self.dataset).__name__)
            self.func = None

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=int(opt.num_threads),
            sampler=self.distributed_sampler,
            worker_init_fn=self.func
        )

        # if self.opt.prefetch:
        #     self.dataloader = DataPrefetcher(self.dataloader)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

        # if not self.opt.prefetch:
        #     for i, data in enumerate(self.dataloader):
        #         if i * self.opt.batch_size >= self.opt.max_dataset_size:
        #             break
        #         yield data
        # else:
        #     self.data = self.dataloader.next()
        #     self.i = 0
        #     while self.data is not None:
        #         if self.i * self.opt.batch_size >= self.opt.max_dataset_size:
        #             break
        #         yield self.data
        #         self.data = self.dataloader.next()
        #         self.i+=1


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            # self.next_data_1, self.next_data_2 = next(self.loader)
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            # self.next_data_2 = None
            return
        with torch.cuda.stream(self.stream):
            if type(self.next_data) is dict:
                for k, v in self.next_data.items():
                    self.next_data[k] = v.cuda(non_blocking=True)
            # self.next_data = self.next_data.cuda(non_blocking=True)
            # self.next_data_2 = self.next_data_2.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


