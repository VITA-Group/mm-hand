import os
import pickle
import random
import sys

import numpy as np
import torch
from cv2 import cv2
from easydict import EasyDict as edict
from torch.utils.data import Dataset

PALM_COLOR = [1] * 3
THUMB_COLOR1 = [2] * 3
THUMB_COLOR2 = [3] * 3
THUMB_COLOR3 = [4] * 3
INDEX_COLOR1 = [5] * 3
INDEX_COLOR2 = [6] * 3
INDEX_COLOR3 = [7] * 3
MIDDLE_COLOR1 = [8] * 3
MIDDLE_COLOR2 = [9] * 3
MIDDLE_COLOR3 = [10] * 3
RING_COLOR1 = [11] * 3
RING_COLOR2 = [12] * 3
RING_COLOR3 = [13] * 3
PINKY_COLOR1 = [14] * 3
PINKY_COLOR2 = [15] * 3
PINKY_COLOR3 = [16] * 3


def generate_jointsmap(uv_coord, depth, width, height, channel=3):
    canvas = np.ones((height, width, channel)) * sys.maxsize
    _canvas = canvas.copy()
    bones = [
        ((0, 17), [160] * channel),
        ((0, 1), [170] * channel),
        ((0, 5), [180] * channel),
        ((0, 9), [190] * channel),
        ((0, 13), [200] * channel),
        ((17, 18), [130] * channel),
        ((18, 19), [140] * channel),
        ((19, 20), [150] * channel),
        ((1, 2), [10] * channel),
        ((2, 3), [20] * channel),
        ((3, 4), [30] * channel),
        ((5, 6), [40] * channel),
        ((6, 7), [50] * channel),
        ((7, 8), [60] * channel),
        ((9, 10), [70] * channel),
        ((10, 11), [80] * channel),
        ((11, 12), [90] * channel),
        ((13, 14), [100] * channel),
        ((14, 15), [110] * channel),
        ((15, 16), [120] * channel),
    ]

    for connection, color in bones:
        temp_canvas = np.ones(canvas.shape) * sys.maxsize

        coord1 = uv_coord[connection[0]]
        coord2 = uv_coord[connection[1]]

        coords = np.stack([coord1, coord2])
        avg_depth = (depth[connection[0]] + depth[connection[1]]) / 2
        x = coords[:, 0]
        y = coords[:, 1]
        mX = x.mean()
        mY = y.mean()
        length = ((x[0] - x[1])**2 + (y[0] - y[1])**2)**0.5
        angle = np.math.degrees(np.math.atan2(y[0] - y[1], x[0] - x[1]))
        radius = 5
        polygon = cv2.ellipse2Poly(
            (int(mX), int(mY)), (int(length / 2), radius), int(angle), 0, 360,
            1)
        cv2.fillConvexPoly(temp_canvas, polygon, [avg_depth] * channel)
        _canvas = np.minimum(_canvas, temp_canvas)
        canvas[_canvas == avg_depth] = color[0]
    canvas[canvas == sys.maxsize] = 0
    return canvas


class Genericdataset(Dataset):
    def __init__(self, opt):
        """ 
        """
        super().__init__()
        self.opt = opt

        self.root_dir = self.opt.dataroot
        with open(os.path.join(self.root_dir, "annotation.pickle"),
                  "rb") as handle:
            self.annotations = pickle.load(handle)

        self.image_source = []
        self.image_target = []

    def _get_src_tgt(self, ratio, data, sort_fn=None):
        """
        generate source and target lists for model base on conditions.
        params:
            ratio  : a float between 0-1 that split the dataset into two, one part for training, the
                     other for augmenting.
            data   : a list of all images path
            sort_fn: a function on how to sort the list of images
        returns: 
            source images (list), target images (list)
        """
        assert len(data) > 0
        if sort_fn is not None:
            data.sort(key=lambda x: sort_fn(x))

        selection_mask = np.zeros(len(data), dtype=np.bool)
        sep_pnt = int((1 - ratio) * len(data))
        src, tgt = [], []
        if 'test' in self.root_dir:
            assert not self.opt.isTrain
            tgt = data
        else:
            if self.opt.isTrain:
                selection_mask[int(sep_pnt)::] = True
            else:
                selection_mask[0:int(sep_pnt)] = True
            tgt = [
                data[i] for i, state in enumerate(selection_mask)
                if state == True
            ]
        src = tgt.copy()
        random.shuffle(src)
        return src, tgt

    def __len__(self):
        return len(self.image_source)

    def __getitem__(self, item):
        h_1 = self.image_source[item]
        h_2 = self.image_target[item]

        h1_annos = self.get_labels(h_1)
        h2_annos = self.get_labels(h_2)

        h1_img = self.make_tensor(
            self.normalize(cv2.cvtColor(cv2.imread(h_1), cv2.COLOR_BGR2RGB)))
        h2_img = self.make_tensor(
            self.normalize(cv2.cvtColor(cv2.imread(h_2), cv2.COLOR_BGR2RGB)))

        h1_map = self.get_heatmaps(h1_annos['uv_coord'], h1_img.shape[1::], 6)
        h2_map = self.get_heatmaps(h2_annos['uv_coord'], h2_img.shape[1::], 6)

        h1_depth = cv2.imread(h_1.replace("color", "depth"))
        h2_depth = cv2.imread(h_2.replace("color", "depth"))

        h1_depth = torch.tensor(256.0 * h1_depth[:, :, 1] + h1_depth[:, :, 2])
        # depth = 256 * g + r
        h2_depth = torch.tensor(256.0 * h2_depth[:, :, 1] + h2_depth[:, :, 2])

        h1_depth = (
            (torch.stack([h1_depth, h1_depth, h1_depth]) / 700.0) - 0.5) / 0.5
        # simulate rgb image
        h2_depth = (
            (torch.stack([h2_depth, h2_depth, h2_depth]) / 700.0) - 0.5) / 0.5

        h1_uv = np.array(h1_annos['uv_coord'])
        h1_z = np.expand_dims(np.array(h1_annos['depth']), -1) / 700.0 * 255
        h1_xyz = torch.tensor(np.concatenate([h1_uv, h1_z], axis=-1))

        h2_uv = np.array(h2_annos['uv_coord'])
        h2_z = np.expand_dims(np.array(h2_annos['depth']), -1) / 700.0 * 255
        h2_xyz = torch.tensor(np.concatenate([h2_uv, h2_z], axis=-1))

        batch = {}
        batch['H1'] = h1_img
        batch['H2'] = h2_img
        batch['P1'] = h1_map
        batch['P2'] = h2_map
        batch['D1'] = h1_depth
        batch['D2'] = h2_depth
        batch['C1'] = h1_xyz
        batch['C2'] = h2_xyz
        batch['H1_path'] = h_1
        batch['H2_path'] = h_2
        return batch

    @staticmethod
    def normalize(img):
        """normalize image range  [0-255] to [-1, 1] """
        return ((img / 255.0) - 0.5) / 0.5

    @staticmethod
    def make_tensor(img):
        return torch.tensor(img).permute(2, 0, 1).float()

    def get_heatmaps(self, uv_coords, shape, sigma):
        heatmaps = []
        for x, y in uv_coords:
            heatmaps.append(
                torch.tensor(
                    self.gen_heatmap(x, y, shape, sigma).astype(np.float32)))
        heatmaps = torch.stack(heatmaps)
        heatmaps = heatmaps.squeeze(1)
        return heatmaps

    def get_labels(self, image_path):
        *_, folder, name = image_path.split('/')
        if "joints" in name:
            name = name.split('_')
            name = name[0] + "_" + name[1] + "_" + name[-1]
        return self.annotations[folder][name]

    def gen_heatmap(self, x, y, shape, sigma):
        # base on DGGAN description
        # a heat map is a dirac-delta function on (x,y) with Gaussian Distribution sprinkle on top.

        centermap = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
        center_map = self.gaussian_kernel(shape[0], shape[1], x, y, sigma)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map
        return center_map

    @staticmethod
    def draw(image, uv_coord, bbox=None):
        """
        draw image with uv_coord and an optional bounding box
        :param image:
        :param uv_coord:
        :param bbox:
        :return: image
        """
        for i, p in enumerate(uv_coord):
            x, y = p
            cv2.circle(image, (int(x), int(y)), 2, 255, 1)
            cv2.putText(image, str(i), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
        if bbox is not None:
            cv2.rectangle(image, (bbox[0], bbox[3]), (bbox[1], bbox[2]), 255,
                          2)
        return image

    @staticmethod
    def gaussian_kernel(width, height, x, y, sigma):
        gridy, gridx = np.mgrid[0:height, 0:width]
        D2 = (gridx - x)**2 + (gridy - y)**2
        return np.exp(-D2 / 2.0 / sigma / sigma)
