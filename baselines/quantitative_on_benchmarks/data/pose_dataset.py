from data.base_dataset import BaseDataset
from data.util import fliplr_joints, get_affine_transform, affine_transform
from pycocotools.coco import COCO
import numpy as np
import torch
import os
import json
import cv2
from multiprocessing import Pool
import random
from copy import deepcopy
from torchvision import transforms

SUPPORT_DATASETS = {"coco": 0,
                    "poseTrack": 1,
                    "MPII": 2}


class PoseDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__(opt)

        self.dataset = self._which_data()
        self.loader = self._get_loader()
        self.ann_ids = self._filter_annotations(self.loader.getAnnIds(iscrowd=False))
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]] # for flip test
        # self.ann_ids = self.loader.getAnnIds()

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, item):
        data = {}
        ann = self.loader.loadAnns(self.ann_ids[item])[0]

        flip = 0 if self.opt.no_flip else random.random() < .5

        img_w, img_h = self._get_shape(ann)
        x, y, w, h = self._get_bbox(ann)

        center, scale = self._xywh2cs(x, y, w, h)
        rotation = 0

        data['bbox_shape'] = torch.tensor([w, h])
        data['image_shape'] = torch.tensor([img_w, img_h], dtype=torch.float)

        kp_3d = np.array(ann['keypoints'],dtype=np.float).reshape(self.opt.num_joints, 3)

        xy_coords = kp_3d[:, 0:-1]
        visibility = (kp_3d[:, -1] >= 1).astype(np.int).reshape((self.opt.num_joints, 1))

        if self.opt.isTrain:
            sf = self.opt.scale_f
            rf = self.opt.rotate_f
            scale = scale * np.clip(np.random.rand() * sf + 1, 1-sf, 1 + sf)
            rotation = np.clip(np.random.randn()*rf, -rf*2, rf*2) if random.random() <= 0.6 else 0

            if flip:
                xy_coords, visibility = fliplr_joints(xy_coords, visibility,
                                                           img_w, self.flip_pairs)
                center[0] = img_w - center[0] - 1
                # center[1] = img_h - center[1] - 1
        else:
            flip = False

        # print(f"xy_coords shape: {xy_coords.shape}, visibility: {visibility.shape}")
        # print(f"rotation : {rotation}")
        # print(f"scale    : {scale}")
        # print(f'center   : {center}')
        # print(f"flip     : {flip}")


        trans = get_affine_transform(center,
                                     scale,
                                     rotation,
                                     (self.opt.crop_width, self.opt.crop_height))

        for i in range(self.opt.num_joints):
            if visibility[i] > 0.0:
                xy_coords[i] = affine_transform(xy_coords[i], trans)

        data.update(self._load_data(ann,
                                    xy_coords.copy(),
                                    visibility.copy(),
                                    flip,
                                    trans))
        data['keypoints'] = torch.tensor(xy_coords)
        # path = self._get_image_path(ann)
        # data['true_image'] = img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # data['center'] = center
        # data['bbox'] = self._get_bbox(ann)
        if self.opt.use_mixed_precision and self.opt.opt_level == "O1":
            for k, v in data.items():
                data[k] = v.half()

        return data

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        aspect_ratio = self.opt.crop_width * 1. / self.opt.crop_height
        pixel_std = 200
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _load_data(self, *args):
        data_types = self.opt.included_data.split('_')
        results = {}
        for d in data_types:
            if hasattr(self, f"_get_{d}"):
                results.update(getattr(self, f"_get_{d}")(*args))
            else:
                raise ValueError(f"{d} is not a supported data type")
        return results

    def _get_image(self, *args):
        annotation, *_, flip, trans = args
        path = self._get_image_path(annotation)
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img_w, img_h = self._get_shape(annotation)
        x, y, w, h = [int(i) for i in self._get_bbox(annotation)]
        #img = img[y:y+h, x:x+w]
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)

        if self.opt.apply_mask:
            mask = self._gen_mask(self._get_bbox(annotation), img.shape)
            img = cv2.bitwise_and(img, img, mask=mask)


        if self.opt.input_nc == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            norm = transforms.Normalize(mean=[np.mean([0.485, 0.456, 0.406])], std=[np.mean([0.229, 0.224, 0.225])])
        else:
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        if flip:
            img = cv2.flip(img, 1)

        img = cv2.warpAffine(img,
                             trans,
                             (int(self.opt.crop_width), int(self.opt.crop_height)),

                             flags=cv2.INTER_LINEAR)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        norm]
                                        )

        # img = torch.tensor(img)
        # img = self._normalize(img)
        # img = img.permute(2, 0, 1)
        return {"image": transform(img)}

    def _get_jointsmap(self, *args):
        """ return a stick figure image from body joints location
        xy_coords = [[x, y, viz]] * 17
        size = the size of the image
        """
        ann, xy_coords, visibility, *_ = args

        _, visibility = self.generate_target(xy_coords, visibility)

        height = self.opt.crop_height
        width = self.opt.crop_width
        skeleton = [
            [[16, 14], [10] * 3],
            [[14, 12], [20] * 3],
            [[17, 15], [30] * 3],
            [[15, 13], [40] * 3],

            # torso
            # [[12, 13], [50]*3],
            # [[6, 12],  [60]*3],
            # [[7, 13],  [70]*3],
            # [[6, 7],   [80]*3],

            [[6, 8], [90] * 3],
            [[7, 9], [100] * 3],
            [[8, 10], [110] * 3],
            [[9, 11], [120] * 3],
            [[2, 3], [130] * 3],
            [[1, 2], [140] * 3],
            [[1, 3], [150] * 3],
            [[2, 4], [160] * 3],
            [[3, 5], [170] * 3],]

            # [[4, 6], [180] * 3],
            # [[5, 7], [190] * 3]]

        img = np.zeros((height, width, self.opt.input_nc), dtype=np.int32)

        torso = [12, 13, 7, 6]
        # print([np.array(xy_coords[i-1][0:-1], dtype=np.int32) for i in torso])
        cv2.fillPoly(img, np.array([[xy_coords[i - 1] for i in torso]], dtype=np.int32), [80] * 3)

        for i, (connection, color) in enumerate(skeleton):
            if visibility[connection[0] - 1, -1] == 0 or visibility[connection[1] - 1, -1] == 0:
                continue
            coord1 = xy_coords[connection[0] - 1]
            coord2 = xy_coords[connection[1] - 1]
            coords = np.stack([coord1, coord2])

            if coord1[-1] == 0 or coord2[-1] == 0:
                continue
            x = coords[:, 0]
            y = coords[:, 1]
            mX = x.mean()
            mY = y.mean()
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = np.math.degrees(np.math.atan2(y[0] - y[1], x[0] - x[1]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), 16), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(img, polygon, color)

        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)

        return {'heatmap': img}

    def _get_heatmaps(self, *args):
        ann, xy_coord, visibility, *_ = args
        # visibility = np.array([visibility])
        target, target_weight = self.generate_target(xy_coord, visibility)
        return {'heatmaps': torch.tensor(target),
                "visibility": torch.tensor(target_weight)}
        # heatmaps = []
        # for x, y, viz in xy_coords:
        #     heatmaps.append(torch.tensor(self._gen_heatmap(x, y, flip)))
        # heatmaps = torch.stack(heatmaps)
        # heatmaps = heatmaps.squeeze(1)
        # return heatmaps

    def _get_heatmap(self, *args):
        data = self._get_heatmaps(*args)
        heatmap = torch.unsqueeze(torch.sum(data['heatmaps'], dim=0), dim=0)
        return {"heatmap": heatmap}

    def _get_sticksmap(self, *args):
        ann, xy_coords, visibility, *_ = args

        _, visibility = self.generate_target(xy_coords, visibility)

        height = self.opt.heatmap_height
        width = self.opt.heatmap_width
        skeleton = [
            [[16, 14], [10] * 3],
            [[14, 12], [20] * 3],
            [[17, 15], [30] * 3],
            [[15, 13], [40] * 3],

            [[12, 13], [50] * 3],
            [[6, 12], [60] * 3],
            [[7, 13], [70] * 3],
            [[6, 7], [80] * 3],

            [[6, 8], [90] * 3],
            [[7, 9], [100] * 3],
            [[8, 10], [110] * 3],
            [[9, 11], [120] * 3],
            [[2, 3], [130] * 3],
            [[1, 2], [140] * 3],
            [[1, 3], [150] * 3],
            [[2, 4], [160] * 3],
            [[3, 5], [170] * 3],

            [[1, 6], [180] * 3], # nose to left shoulder
            [[1, 7], [190] * 3]] # nose to right shoulder

            # [[4, 6], [180] * 3],
            # [[5, 7], [190] * 3]]

        img = np.zeros((height, width, self.opt.input_nc), dtype=np.int32)
        for connection, color in skeleton:
            if visibility[connection[0]-1, -1] == 0 or visibility[connection[1]-1, -1] == 0:
                continue
            coord1 = tuple(xy_coords[connection[0] - 1].astype(np.int32).tolist())
            coord2 = tuple(xy_coords[connection[1] - 1].astype(np.int32).tolist())

            img = cv2.line(img, coord1, coord2, color, 2)

        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        return {"heatmap": img}

    def _gen_heatmap(self, x, y, flip):
        centermap = np.zeros((self.opt.heatmap_height, self.opt.heatmap_width, 1), dtype=np.float32)
        center_map = self._gaussian_kernel(self.opt.heatmap_width,
                                           self.opt.heatmap_height, x, y, 2)
        # print(center_map.shape)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map
        if flip:
            center_map = cv2.flip(center_map, 1)
        return center_map

    def _gen_mask(self, bbox, shape):
        """

        :param bbox: x, y, w, h
        :param shape: shape of image
        :return:
        """
        mask = np.zeros(shape[0:-1], dtype=np.uint8)
        x,y,w,h = bbox
        p1 = x, y
        p2 = x+w, y
        p3 = x+w, y+h
        p4 = x, y+h
        mask = cv2.fillPoly(mask, np.array([[p1, p2, p3, p4]], dtype=np.int32), 255)
        return mask

    def _which_data(self):
        global SUPPORT_DATASETS
        dataset = os.path.basename(self.root)
        # print(dataset)
        if dataset not in SUPPORT_DATASETS:
            raise TypeError("not a supported dataset")
        else:
            return SUPPORT_DATASETS[dataset]

    def _get_loader(self):
        funcs = [self._load_coco, self._load_posetrack, self._load_mpii]
        return funcs[self.dataset]()

    def _load_coco(self):
        ltype = 'train' if self.opt.isTrain else 'val'
        path = f"{self.root}/annotations/person_keypoints_{ltype}2017.json"
        return COCO(path)
        # with open(path, "r") as f:
        #     return json.load(f), COCO(path)

    def _load_mpii(self):
        # if self.opt.isTrain:
        #     pass
        # else:
        #     pass
        raise NotImplemented

    def _load_posetrack(self):
        ltype = 'train' if self.opt.isTrain else 'test'
        path = f"{self.root}/annotations/{ltype}/{ltype}.json"
        return COCO(path)
        # with open(path, 'r') as f:
        #     return json.load(f), COCO(path)

    def _filter_annotations(self, annotations):
        '''
        'keypoints': [ 'nose',           1
                       'head_bottom',    2
                       'head_top',       3
                       'left_ear',       4
                       'right_ear',      5
                       'left_shoulder',  6
                       'right_shoulder', 7
                       'left_elbow',     8
                       'right_elbow',    9
                       'left_wrist',    10
                       'right_wrist',   11
                       'left_hip',      12
                       'right_hip',     13
                       'left_knee',     14
                       'right_knee',    15
                       'left_ankle',    16
                       'right_ankle'],  17
        '''
        # from tqdm import tqdm
        # if not self.opt.distributed or self.opt.distributed and self.opt.local_rank == 0:
        #     args = [[self.loader.loadAnns(i)[0], self.opt.threshold] for i in annotations]
        #     with Pool() as p:
        #         result = list(tqdm(p.imap(self._filterworker, args), total=len(annotations)))
        #         return [i for i, r in zip(annotations, result) if r]
        # else:
        args = []
        for i in annotations:
            anns = self.loader.loadAnns(i)[0]
            img_ann = self.loader.loadImgs(anns['image_id'])[0]
            args.append([anns, img_ann, self.opt.threshold])
        # args = [[self.loader.loadAnns(i)[0], self.opt.threshold] for i in annotations]
        with Pool() as p:
            result = list(p.imap(self._filterworker, args))
            return [i for i, r in zip(annotations, result) if r]

    @staticmethod
    def _normalize(image):
        # height, width, channel = image.shape
        # norm_image = cv2.normalize(image, None, 0, 1, norm_type=cv2.NORM_MINMAX,
        #                            dtype=cv2.CV_32F)
        # return norm_image
        pass

    @staticmethod
    def _gaussian_kernel(height, width, x, y, sigma):
        gridy, gridx = np.mgrid[0:width, 0:height]
        D2 = (gridx - x) ** 2 + (gridy - y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)

    def _get_image_path(self, annotation):
        meta = self.loader.loadImgs(annotation['image_id'])[0]
        if self.dataset == 0:
            ltype = 'train' if self.opt.isTrain else 'val'
            path = f"{self.root}/images/{ltype}2017/{meta['file_name']}"
        elif self.dataset == 1:
            path = f"{self.root}/{meta['file_name']}"
        else:
            raise NotImplemented

        return path

    @staticmethod
    def _filterworker(*args):
        ann, img_ann, thres= args[0]
        kp = np.array(ann['keypoints']).reshape((17, 3))
        width, height = img_ann['width'], img_ann['height']
        x, y, w, h = ann['bbox']
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        if ann['area'] == 0 or x2 < x1 or y2 < y1:
            return False
        return kp[:, -1].sum() > 0

    def _get_shape(self, annotation):
        img_ann = self.loader.loadImgs(annotation['image_id'])[0]
        return float(img_ann['width']), float(img_ann['height'])

    def _get_bbox(self, ann):
        width, height = self._get_shape(ann)
        x, y, w, h = ann['bbox']
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        return x1, y1, x2-x1, y2-y1

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.opt.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((self.opt.num_joints,
                           self.opt.heatmap_height,
                           self.opt.heatmap_width),
                          dtype=np.float32)
        tmp_size = self.opt.sigma * 3
        image_size = np.array([self.opt.crop_width, self.opt.crop_height], dtype=np.int)
        heatmap_size = np.array([self.opt.heatmap_width, self.opt.heatmap_height], dtype=np.int)


        for joint_id in range(self.opt.num_joints):
            feat_stride = image_size / heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.opt.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target, target_weight

