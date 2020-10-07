from utils import Evaluator
from networks.net_resnetgenerator import ResnetGenerator as G
from easydict import EasyDict as edict
from tqdm import tqdm
from data.STB_dataset import STBdataset
from data.RHD_dataset import RHDdataset
import torch
if __name__ == "__main__":
    norm_layer = torch.nn.modules.instancenorm.InstanceNorm2d
    model = G(3, 3, 64,norm_layer=norm_layer , n_blocks=9)
    device = 0

    weights = "./weights/cyclegan/stb_net_G.pth"

    cpm2d = './weights/cpm/stb_hpm2d.pth'
    cpm3d = './weights/cpm/stb_hpm3d.pth'
    evaluate = Evaluator(model, weights, cpm2d, cpm3d, device)
    opt = edict()
    opt.dataroot = "./dataset/stb-dataset/test"
    opt.isTrain = False
    dataset = STBdataset(opt)
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        evaluate.feed(sample['xyz'], sample['A']*255.0)
    results = evaluate.evaluate()
    print(f"STB results: {results}")

    opt.dataroot = "./dataset/rhd-dataset/test"
    weights = "./weights/cyclegan/rhd_net_G.pth"
    cpm2d = "./weights/cpm/rhd_hpm2d.pth"
    dataset = RHDdataset(opt)
    evaluate = Evaluator(model, weights, cpm2d, cpm3d, device)
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        evaluate.feed(sample['xyz'], sample['A']*255.0)
    results = evaluate.evaluate()
    print(f"rhd results: {results}")
