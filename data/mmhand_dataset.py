import os.path
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from data.image_folder import make_dataset


class MMHandDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dir_H = os.path.join(opt.imageroot, opt.phase)  # hand images
        self.dir_P = os.path.join(opt.poseroot, opt.phase + 'P')  # hand poses

        self.init_categories(opt.pairLst)
        self.transform = self.get_transform()

    def init_categories(self, pairLst):

        pairs_file = pd.read_csv(pairLst)
        self.size = len(pairs_file)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file.iloc[i]['from'], pairs_file.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def get_transform(self):
        transform_list = []

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size - 1)

        H1_name, H2_name = self.pairs[index]
        '''Added for Depth Map'''
        # hand 1 and its pose and depth
        if '.png' not in H1_name:
            H1_path = os.path.join(self.dir_H, H1_name + '.png')
        else:
            H1_path = os.path.join(self.dir_H, H1_name)
        P1_path = os.path.join(self.dir_P,
                               H1_name + '.npy')  # bone of person 1
        D1_path = H1_path.replace('color', 'depth')

        # hand 2 and its pose and depth
        if '.png' not in H2_name:
            H2_path = os.path.join(self.dir_H, H2_name + '.png')  # person 2
        else:
            H2_path = os.path.join(self.dir_H, H2_name)  # person 2
        P2_path = os.path.join(self.dir_P,
                               H2_name + '.npy')  # bone of person 2
        D2_path = H2_path.replace('color', 'depth')

        H1_img = Image.open(H1_path).convert('RGB')
        H2_img = Image.open(H2_path).convert('RGB')

        P1_img = np.load(P1_path)  # h, w, c
        P2_img = np.load(P2_path)

        D1_img = Image.open(D1_path).convert('RGB')
        D2_img = Image.open(D2_path).convert('RGB')

        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip:
            # print ('use_flip ...')
            flip_random = random.uniform(0, 1)

            if flip_random > 0.5:
                # print('fliped ...')
                H1_img = H1_img.transpose(Image.FLIP_LEFT_RIGHT)
                H2_img = H2_img.transpose(Image.FLIP_LEFT_RIGHT)

                P1_img = np.array(P1_img[:, ::-1, :])  # flip
                P2_img = np.array(P2_img[:, ::-1, :])  # flip

                D1_img = D1_img.transpose(Image.FLIP_LEFT_RIGHT)
                D2_img = D2_img.transpose(Image.FLIP_LEFT_RIGHT)

        P1 = torch.from_numpy(P1_img).float()  #h, w, c
        P1 = P1.transpose(2, 0)  #c,w,h
        P1 = P1.transpose(2, 1)  #c,h,w

        P2 = torch.from_numpy(P2_img).float()
        P2 = P2.transpose(2, 0)  #c,w,h
        P2 = P2.transpose(2, 1)  #c,h,w

        H1 = self.transform(H1_img)
        H2 = self.transform(H2_img)
        '''Added for Depth Map'''
        D1 = self.transform(D1_img)
        D2 = self.transform(D2_img)
        return {
            'H1': H1,
            'P1': P1,
            'D1': D1,
            'H1_path': H1_name,
            'H2': H2,
            'P2': P2,
            'D2': D2,
            'H2_path': H2_name
        }

    def __len__(self):
        return self.size
