from utils import Evaluator
from networks.net_unetgenerator import UnetGenerator as G
from easydict import EasyDict as edict
from tqdm import tqdm
from data.STB_dataset import STBdataset
from data.RHD_dataset import RHDdataset


if __name__ == "__main__":
    model = G(3, 3, 8)
    device = 0
    cpm2d = './weights/cpm/stb_hpm2d.pth'
    cpm3d = './weights/cpm/stb_hpm3d.pth'
    weights = "./weights/pix2pix/stb_net_G.pth"
    evaluate = Evaluator(model, weights,cpm2d, cpm3d, device)
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
    cpm2d = "./weights/cpm/rhd_hpm2d.pth"
    weights = "./weights/pix2pix/stb_net_G.pth"
    dataset = RHDdataset(opt)
    evaluate = Evaluator(model, weights, cpm2d, cpm3d, device)
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        evaluate.feed(sample['xyz'], sample['A']*255.0)
    results = evaluate.evaluate()
    print(f"rhd results: {results}")
