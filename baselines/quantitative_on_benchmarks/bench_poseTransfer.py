from utils import Evaluator
from networks.model_variants import PATNetwork as G
from easydict import EasyDict as edict
from tqdm import tqdm
from data.STB_dataset_pose_trasfer import STBdataset
from data.RHD_dataset_pose_trasfer import RHDdataset
import torch
Tensor = torch.Tensor

def set_input(sample, netG, device):
    with torch.no_grad():
        nb = 1
        size = 256
        input_P1_set =  Tensor(nb, 3, size, size)
        input_BP1_set =  Tensor(nb, 21, size, size)
        input_P2_set =  Tensor(nb, 3, size, size)
        input_BP2_set =  Tensor(nb, 21, size, size)

        input_P1, input_BP1 = sample['P1'].unsqueeze(0), sample['BP1'].unsqueeze(0)
        input_P2, input_BP2 = sample['P2'].unsqueeze(0), sample['BP2'].unsqueeze(0)

        input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        input_BP1_set.resize_(input_BP1.size()).copy_(input_BP1)
        input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        input_BP2_set.resize_(input_BP2.size()).copy_(input_BP2)


        input_P1 = input_P1_set
        input_BP1 = input_BP1_set

        input_P2 = input_P2_set
        input_BP2 = input_BP2_set

        if device != 'cpu':
            input_P1 = input_P1.to(device)
            input_BP1 = input_BP1.to(device)
            input_BP2 = input_BP2.to(device)
            input_P2 = input_P2.to(device)

        G_input = [input_P1,
                    torch.cat((input_BP1, input_BP2), 1)]
        fake_p2 = netG(G_input)
    return fake_p2, input_P2 * 255.0, sample['P2xyz']

if __name__ == "__main__":
    model = G([3,42], 3, 64, torch.nn.BatchNorm2d, True, 9, [], 'reflect', 2)
    w = "./weights/pose-transfer/stb_net_netG.pth"
    cpm2d = './weights/cpm/stb_hpm2d.pth'
    cpm3d = './weights/cpm/stb_hpm3d.pth'
    device = 0
    evaluate = Evaluator(model, w,cpm2d, cpm3d, device)
    opt = edict()
    opt.dataroot = './dataset/stb-dataset/test'
    opt.isTrain = False
    dataset = STBdataset(opt)
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        pred, gt, xyz = set_input(sample, evaluate.model, evaluate.device)
        evaluate._feed(pred, gt, xyz)
    print(f"stb results: {evaluate.evaluate()}")

    opt.dataroot = './dataset/rhd-dataset/test'
    w ="./weights/pose-transfer/rhd_net_netG.pth"
    cpm2d = "./weights/cpm/rhd_hpm2d.pth"
    evaluate = Evaluator(model, w, cpm2d, cpm3d, device)
    opt.isTrain = False
    dataset = RHDdataset(opt)
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        pred, gt, xyz = set_input(sample, evaluate.model, evaluate.device)
        evaluate._feed(pred, gt, xyz)
    print(f"rhd results: {evaluate.evaluate()}")
