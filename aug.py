import os
import sys

import cv2
import torch
import tqdm
from data.mmhand_dataset_data_loader import MMHandDatasetDataLoader
from easydict import EasyDict as edict

from models.Generator import Generator as G
from models.network_utils import get_norm_layer
if __name__ == "__main__":

    _, ckp, dataroot, DST, dataset, ratio, device = sys.argv
    device = int(device)
    opt = edict()
    opt.dataroot = dataroot
    opt.isTrain = False
    opt.dataset = dataset
    opt.augmentation_ratio = float(ratio)
    opt.distributed = False
    opt.batchSize = 1
    opt.nThreads = 4

    dataloader = MMHandDatasetDataLoader(opt)
    weights = torch.load(
        os.path.join('checkpoints', ckp, 'latest_net_netG.pth'), 'cpu')

    input_nc = [3, 42, 6]
    norm = get_norm_layer('batch')
    model = G(input_nc=input_nc,
              output_nc=3,
              ngf=64,
              norm_layer=norm,
              use_dropout=True,
              n_blocks=9)
    model.load_state_dict(weights)
    model = model.to(device)
    model = model.eval()

    if not os.path.isdir(DST): os.mkdir(DST)
    for i, sample in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        input_P1 = sample['P1'].to(device).float()
        input_P2 = sample['P2'].to(device).float()
        input_D1 = sample['D1'].to(device).float()
        input_D2 = sample['D2'].to(device).float()
        input_H1 = sample['H1'].to(device).float()

        model_input = [
            input_H1,
            torch.cat((input_P1, input_P2), 1),
            torch.cat((input_D1, input_D2), 1)
        ]

        fake = None
        real = None
        with torch.no_grad():
            fake_p2 = model(model_input)
            fake = fake_p2.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            fake = (fake * 0.5 + 0.5) * 255.
            fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)

            real = sample['H2'].squeeze(0).permute(1, 2, 0).cpu().numpy()
            real = (real * 0.5 + 0.5) * 255.
            real = cv2.cvtColor(real, cv2.COLOR_RGB2BGR)

        *_, folder, name = sample['H2_path'][0].split('/')
        dst_i = os.path.join(DST, folder)
        if not os.path.isdir(dst_i): os.mkdir(dst_i)

        cv2.imwrite(os.path.join(dst_i, name), fake)
