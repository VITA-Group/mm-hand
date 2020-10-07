import numpy as np
import torch
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from data.msrahand_dataset import MSRAHandDataset as msra
import tqdm
import cv2
import math
from kdtree import create

class Item:
    def __init__(self, xyz, uv, index):
        self.index = index
        self.xyz = xyz
        self.uv = uv
        centroid = self._get_centroid(xyz)

        area = self._get_area(uv)

        tip_distances = self._get_tip_distances(xyz)

        self.dimension = [*centroid, *tip_distances, math.sqrt(area)]

    def __len__(self):
        return len(self.dimension)

    def __getitem__(self, i):
        return self.dimension[i]

    def __repr__(self):
        return str(self.index)

    @staticmethod
    def _get_centroid(coords):
        coord_t = np.array(coords).T
        cx, cy, cz = np.sum(coord_t, axis=1) / coord_t.shape[1]
        return cx, cy, cz

    @staticmethod
    def _get_area(coords):
        convex_hull = cv2.convexHull(coords)
        return cv2.contourArea(convex_hull)

    def _get_tip_distances(self, coords):
        # vary by dataset, this one works on MSRA
        #         thumbt = coords[20]
        #         indext = coords[4]
        #         midt = coords[8]
        #         ringt = coords[12]
        #         pinkyt = coords[16]
        palm = coords[0]
        d = []
        fingers = [coords[20],
                   coords[4],
                   coords[8],
                   coords[12],
                   coords[16]]
        for tip in fingers:
            d.append(self._get_distance(palm, tip))

        return d

    @staticmethod
    def _get_distance(c0, c1):
        return np.linalg.norm(c0 - c1)


def poseDistance(u, v):
    def identity(u):
        u = np.squeeze(u)
        assert (u.shape[0] == 21 and u.shape[1] == 3)
        u_i = []
        for i in range(1, 21):
            prev = u[i - 1]
            cur = u[i]
            u_i.append(cur - prev)
        u_i = np.array(u_i)
        u_i.resize(u_i.shape[0] * u_i.shape[1])
        return u_i
    u_i = identity(u)
    v_i = identity(v)
    d = 1./np.pi * np.arccos(np.dot(u_i, v_i) / (np.linalg.norm(u_i) * np.linalg.norm(v_i)))
    return d

def wrapper_poseDistance(item0, item1):
    u = item0.xyz
    v = item1.xyz
    return poseDistance(u, v)

opt = edict()
opt.dataroot = '../datasets/legacy_datasets/msra_hand'
opt.isTrain = False
dataset = msra(opt)

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=False, sampler=torch.utils.data.SequentialSampler(dataset),
                                       num_workers=24)
c = 0
items = []
for sample in tqdm.tqdm(dataloader, total=len(dataset)//100):
    xyzs = sample['xyz'].numpy()
    uvs = sample['uv'].numpy()
    for xyz, uv in zip(xyzs, uvs):
        items.append(Item(xyz, uv, c))
        c+=1

tree = create(items)
print(tree.is_balanced)

sim = tree.search_knn(items[549], k=25, dist=wrapper_poseDistance)
nearest_idx = [int(i[0].data.index) for i in sim]
fig = plt.figure(figsize=(50, 50))
col = 1
row = len(nearest_idx)
for i in range(len(nearest_idx)):
    idx = nearest_idx[i]
    image = torch.squeeze(dataset[idx]['depthmap']).numpy()
    fig.add_subplot(row, col, i+1)
    plt.imshow(image)
plt.show()