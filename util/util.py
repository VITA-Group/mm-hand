from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import cv2
import math
from skimage.draw import circle, line_aa, polygon

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


PALM_COLOR = [1]*3
THUMB_COLOR1 = [2]*3
THUMB_COLOR2 = [3]*3
THUMB_COLOR3 = [4]*3
INDEX_COLOR1 = [5]*3
INDEX_COLOR2 = [6]*3
INDEX_COLOR3 = [7]*3
MIDDLE_COLOR1 = [8]*3
MIDDLE_COLOR2 = [9]*3
MIDDLE_COLOR3 = [10]*3
RING_COLOR1 = [11]*3
RING_COLOR2 = [12]*3
RING_COLOR3 = [13]*3
PINKY_COLOR1 = [14]*3
PINKY_COLOR2 = [15]*3
PINKY_COLOR3 = [16]*3

# draw pose img
# LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
#            [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
#            [0,15], [15,17], [2,16], [5,17]]
LIMB_SEQ = [[0,1], [1,2], [2,3], [3,4],
            [0,5], [5,6], [6,7], [7,8],
            [0,9], [9,10], [10,11], [11,12],
            [0,13], [13,14], [14,15], [15,16],
            [0,17], [17,18], [18,19], [19,20]]

BONES = [((1, 2), THUMB_COLOR1),
         ((2, 3), THUMB_COLOR2),
         ((3, 4), THUMB_COLOR3),

         ((5, 6), INDEX_COLOR1),
         ((6, 7), INDEX_COLOR2),
         ((7, 8), INDEX_COLOR3),

         ((9, 10), MIDDLE_COLOR1),
         ((10, 11), MIDDLE_COLOR2),
         ((11, 12), MIDDLE_COLOR3),

         ((13, 14), RING_COLOR1),
         ((14, 15), RING_COLOR2),
         ((15, 16), RING_COLOR3),

         ((17, 18), PINKY_COLOR1),
         ((18, 19), PINKY_COLOR2),
         ((19, 20), PINKY_COLOR3)]

PALM = [((0, 1), PALM_COLOR),
        ((1, 5), PALM_COLOR),
        ((5, 9), PALM_COLOR),
        ((9, 13), PALM_COLOR),
        ((13, 17), PALM_COLOR),
        ((17, 0), PALM_COLOR)
        ]
# COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
#           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
#           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# COLORS = [[255, 255, 255],
#           [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0],
#           [170, 255, 0], [170, 255, 0], [170, 255, 0], [170, 255, 0],
#           [0, 255, 170], [0, 255, 170], [0, 255, 170], [0, 255, 170],
#           [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255],
#           [255, 0, 170], [255, 0, 170], [255, 0, 170], [255, 0, 170]]


# LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
#                'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1

def map_to_cord(pose_map, threshold=0.1):
    all_peaks = [[] for i in range(21)]
    pose_map = pose_map[..., :21]

    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis = (0, 1)),
                                     pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(21):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)

def draw_pose_from_map(pose_map, threshold=0.1, **kwargs):
    # CHW -> HCW -> HWC
    pose_map = pose_map[0].cpu().transpose(1, 0).transpose(2, 1).numpy()

    cords = map_to_cord(pose_map, threshold=threshold)
    return draw_pose_from_cords(cords, pose_map.shape[:2], **kwargs)

class Colorize(object):
    def __init__(self, n):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def add_color(self, gray_image):
        size = gray_image.shape
        color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)
        for label in range(self.cmap.shape[0]):
            mask = (gray_image == label)
            #print(mask.shape)
            mask = torch.from_numpy(np.array(mask, dtype=np.bool))
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def labelcolormap(N):
    def uint82bin(n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


# draw pose from map
def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    canvas = np.zeros(shape=img_size + (3, ), dtype=np.uint8)

    palm = []
    for connection, _ in PALM:
        coord = pose_joints[connection[0]]
        palm.append([coord[1], coord[0]])

    cv2.fillConvexPoly(canvas,np.array( [palm], dtype=np.int32 ),PALM_COLOR)

    for connection, color in BONES:
        coord1 = pose_joints[connection[0]]
        coord2 = pose_joints[connection[1]]
        coords = np.stack([coord1, coord2])
        x = coords[:, 1]
        y = coords[:, 0]
        mX = x.mean()
        mY = y.mean()
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), 8), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, color)

    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    label_tensor = Colorize(22).add_color(canvas)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_numpy.astype(np.uint8)
    return label_numpy

# # draw pose from map
# def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
#     colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
#     mask = np.zeros(shape=img_size, dtype=bool)
#
#     if draw_joints:
#         for f, t in LIMB_SEQ:
#             from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
#             to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
#             if from_missing or to_missing:
#                 continue
#             yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
#             colors[yy, xx] = np.expand_dims(val, 1) * 255
#             mask[yy, xx] = True
#
#     for i, joint in enumerate(pose_joints):
#         if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
#             continue
#         yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
#         colors[yy, xx] = COLORS[i]
#         mask[yy, xx] = True
#
#     return colors, mask


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
