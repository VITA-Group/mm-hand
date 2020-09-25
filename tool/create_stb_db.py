#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pickle
from multiprocessing.pool import Pool

import numpy as np
import torch
import tqdm as tqdm
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os
import cv2
from easydict import EasyDict as edict
import numpy as np
import sys

DEBUG = False
# setting 'borrow' from https://github.com/spurra/vae-hands-3d/blob/master/data/stb/create_db.m
# intrinsic camera values for BB
I_BB = edict()
I_BB.fx = 822.79041
I_BB.fy = 822.79041
I_BB.tx = 318.47345
I_BB.ty = 250.31296
I_BB.base = 120.054
I_BB.R_l = np.array([[0.0, 0.0, 0.0]])
I_BB.R_r = I_BB.R_l
I_BB.T_l = np.array([0.0, 0.0, 0.0])
I_BB.T_r = np.array([-I_BB.base, 0, 0])
I_BB.K = np.diag([I_BB.fx, I_BB.fy, 1.0])
I_BB.K[0, 2] = I_BB.tx
I_BB.K[1, 2] = I_BB.ty

# intrinsic camerae value for SK
I_SK = edict()
I_SK.fx_color = 607.92271
I_SK.fy_color = 607.88192
I_SK.tx_color = 314.78337
I_SK.ty_color = 236.42484
I_SK.K_color = np.diag([I_SK.fx_color, I_SK.fy_color, 1])
I_SK.K_color[0, 2] = I_SK.tx_color
I_SK.K_color[1, 2] = I_SK.ty_color

I_SK.fx_depth = 475.62768
I_SK.fy_depth = 474.77709
I_SK.tx_depth = 336.41179
I_SK.ty_depth = 238.77962
I_SK.K_depth = np.diag([I_SK.fx_depth, I_SK.fy_depth, 1])
I_SK.K_depth[0, 2] = I_SK.tx_depth
I_SK.K_depth[1, 2] = I_SK.ty_depth

I_SK.R_depth = I_BB.R_l.copy()
I_SK.T_depth = I_BB.T_l.copy()
# https://github.com/zhjwustc/icip17_stereo_hand_pose_dataset claims that R and T is for color -> depth trans. It is not.
# it is in fact depth -> color.
I_SK.R_color = -1 * np.array([[0.00531, -0.01196, 0.00301]])
I_SK.T_color = -1 * np.array([-24.0381, -0.4563, -1.2326])

PALM_COLOR = [10] * 3
THUMB_COLOR1 = [20] * 3
THUMB_COLOR2 = [30] * 3
THUMB_COLOR3 = [40] * 3
INDEX_COLOR1 = [50] * 3
INDEX_COLOR2 = [60] * 3
INDEX_COLOR3 = [70] * 3
MIDDLE_COLOR1 = [80] * 3
MIDDLE_COLOR2 = [90] * 3
MIDDLE_COLOR3 = [100] * 3
RING_COLOR1 = [110] * 3
RING_COLOR2 = [120] * 3
RING_COLOR3 = [130] * 3
PINKY_COLOR1 = [140] * 3
PINKY_COLOR2 = [150] * 3
PINKY_COLOR3 = [160] * 3

#
# ordering: palm center(not wrist or hand center), little_mcp, little_pip, little_dip, little_tip, ring_mcp, ring_pip,
# ring_dip, ring_tip, middle_mcp, middle_pip, middle_dip, middle_tip, index_mcp, index_pip, index_dip, index_tip,
# thumb_mcp, thumb_pip, thumb_dip, thumb_tip.

# remapping labels to fit with standard labeling.
STB_TO_STD = [0, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4]


def create_jointsmap(uv_coord, size):
    """ Plots a hand stick figure into a matplotlib figure. """

    # define connections and colors of the bones
    # print(coords_hw[-1]) # this is center ( the 22nd point)
    canvas = np.zeros((size, size, 3))
    bones = [
        ((1, 2), THUMB_COLOR1),
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
    palm = []
    for connection, _ in [((0, 1), []),
                          ((1, 5), []),
                          ((5, 9), []),
                          ((9, 13), []),
                          ((13, 17), []),
                          ((17, 0), []), ]:
        coord1 = uv_coord[connection[0]]
        palm.append([int(coord1[0]), int(coord1[1])])
    # palm.append([int((coord1[0]-.5)* W_scale+ W_offset ), int(-(coord1[1]- .5)* H_scale+ H_offset)])
    # print(palm)
    cv2.fillConvexPoly(canvas, np.array([palm], dtype=np.int32), PALM_COLOR)
    for connection, color in bones:
        coord1 = uv_coord[connection[0]]
        coord2 = uv_coord[connection[1]]
        coords = np.stack([coord1, coord2])
        # 0.5, 0.5 is the center
        x = coords[:, 0]
        y = coords[:, 1]
        mX = x.mean()
        mY = y.mean()
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = np.math.degrees(np.math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), 16), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, color)
    return canvas


def reorder(xyz_coord):
    return xyz_coord[STB_TO_STD]


def get_xyz_coord(path):
    """
    get xyz coordinate from STB's mat file return 1500x21x3 matrix
    :param path:
    :return: hand labels
    """
    labels = loadmat(path)
    anno_xyz = []
    for index in range(0, 1500):
        anno_xyz.append([])
        for i in range(0, 21):
            x = labels['handPara'][0][i][index]
            y = labels['handPara'][1][i][index]
            z = labels['handPara'][2][i][index]
            anno_xyz[-1].append([x, y, z])
    anno_xyz = np.array(anno_xyz)
    # anno_xyz = np.reshape(labels['handPara'], (1500, 21, 3))
    return anno_xyz


def get_uv_coord(mode, camera, anno_xyz):
    """
    gets uv coordinates from xyz coordinate for STB dataset
    :param mode: have to be either "l" for left hand or "r" for right hand. 'c' for color, 'd' for depth.
    :param camera: either "BB" or "SK"
    :param anno_xyz: the 3d coordinate
    :return: uv_coords
    """
    if camera == 'SK':
        # SK only have left hand. this is only for color image. Unable to translate kp to depth image.
        if mode == 'color':
            uv_coord, _ = cv2.projectPoints(anno_xyz, I_SK.R_color, I_SK.T_color, I_SK.K_color, None)
        elif mode == 'depth':
            uv_coord, _ = cv2.projectPoints(anno_xyz, I_SK.R_depth, I_SK.T_depth, I_SK.K_depth, None)
        else:
            raise ValueError
    elif camera == 'BB':
        if mode == 'left':
            uv_coord, _ = cv2.projectPoints(anno_xyz, I_BB.R_l, I_BB.T_l, I_BB.K, None)
        elif mode == 'right':
            uv_coord, _ = cv2.projectPoints(anno_xyz, I_BB.R_r, I_BB.T_r, I_BB.K, None)
        else:
            raise ValueError
    else:
        raise ValueError
    return np.reshape(uv_coord, (21, 2))


def get_bounding_box(uv_coor, shape):
    """
    returns bounding box given 2d coordinate
    :param uv_coor: x,y dataset of joints
    :param shape: height and width of an image
    :return: bounding box
    """
    xmin = ymin = 99999
    xmax = ymax = 0
    for x, y in uv_coor:
        xmin = min(xmin, int(x))
        xmax = max(xmax, int(x))
        ymin = min(ymin, int(y))
        ymax = max(ymax, int(y))
    xmin = max(0, xmin - 20)
    ymin = max(0, ymin - 20)

    xmax = min(shape[1], xmax + 20)
    ymax = min(shape[0], ymax + 20)

    return xmin, xmax, ymin, ymax


def scale(uv_coord, K, bbox, new_size):
    """
    scale and translate key points/K map to new size
    :param uv_coord: 2d key points coordinates
    :param K: Intrinsic matrix
    :param bbox: bounding box of the hand
    :param new_size: new size (width x height)
    :return: uv_coord, K
    """
    xmin, xmax, ymin, ymax = bbox

    uv_coord[:, 0] = (uv_coord[:, 0] - xmin) / (xmax - xmin + 1.) * new_size[1]
    uv_coord[:, 1] = (uv_coord[:, 1] - ymin) / (ymax - ymin + 1.) * new_size[0]

    xscale = new_size[1] / (xmax - xmin + 1.)
    yscale = new_size[0] / (ymax - ymin + 1.)

    shift = [[1, 0, -xmin],
             [0, 1, -ymin],
             [0, 0, 1]]

    scale = [[xscale, 0, 0],
             [0, yscale, 0],
             [0, 0, 1]]

    shift = np.array(shift)
    scale = np.array(scale)

    K = np.matmul(scale, np.matmul(shift, K))

    return uv_coord, K


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
        cv2.circle(image, (int(x), int(y)), 10, 255, 2)
        cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
    if bbox is not None:
        cv2.rectangle(image, (bbox[0], bbox[3]), (bbox[1], bbox[2]), 255, 2)
    return image


def to_tensor(image):
    shape = image.shape
    if shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
    else:
        # grayscale
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
    return image


def get_heatmaps(uv_coords, shape):
    heatmaps = []
    for x, y in uv_coords:
        heatmaps.append(to_tensor(gen_heatmap(x, y, shape).astype(np.float32)))
    heatmaps = torch.stack(heatmaps)
    heatmaps = heatmaps.squeeze(1)
    return heatmaps


def gen_heatmap(x, y, shape):
    # base on DGGAN description
    # a heat map is a dirac-delta function on (x,y) with Gaussian Distribution sprinkle on top.
    centermap = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
    center_map = gaussian_kernel(shape[0], shape[1], x, y, 3)
    center_map[center_map > 1] = 1
    center_map[center_map < 0.0099] = 0
    centermap[:, :, 0] = center_map
    return center_map


def gaussian_kernel(width, height, x, y, sigma):
    gridy, gridx = np.mgrid[0:height, 0:width]
    D2 = (gridx - x) ** 2 + (gridy - y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def image_process(arg):
    img_path, destination, anno_xyz, size = arg
    image = cv2.imread(img_path)
    camera, mode, index = os.path.basename(img_path).split("_")
    depth = anno_xyz[:, -1].copy()
    uv_coor = get_uv_coord(mode, camera, anno_xyz)
    bbox = get_bounding_box(uv_coor, image.shape)

    xmin, xmax, ymin, ymax = bbox
    image = image[ymin:ymax + 1, xmin:xmax + 1]  # crop the image
    image = cv2.resize(image, (size, size))

    if camera == "BB":
        K = I_BB.K.copy()
    else:
        if mode == "color":
            K = I_SK.K_depth.copy()
        else:
            K = I_SK.K_color.copy()
    uv_coor, k = scale(uv_coor, K, bbox, (size, size))

    joints_map = create_jointsmap(uv_coor, size)
    joints_map_name = os.path.basename(destination).split('_')
    joints_map_name = joints_map_name[0] + '_' + joints_map_name[1] + '_' + "joints" + "_" + joints_map_name[2]
    joints_map_path = os.path.join(os.path.dirname(destination), joints_map_name)

    #cv2.imwrite(joints_map_path, joints_map)

    # saving 21x1x256x256 heatmaps as .pt
    heat_maps = get_heatmaps(uv_coor, (size, size))
    heat_maps_path = os.path.join(os.path.dirname(destination), os.path.basename(destination)[0:-3]+"pt")
    #torch.save(heat_maps, heat_maps_path)
    
    if 'depth' in os.path.basename(img_path):
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        image = r + g * 256.0
        norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = norm_image*255

    cv2.imwrite(destination, image)


    return [destination, joints_map_path, heat_maps_path, anno_xyz, uv_coor, depth, k]


def main(src, dst, size):
    """
    run STB preprocessing. which will create a new STB_crop folder where the hand region occupied the majority of the frame.
    replace multiple .mat label files with a single pickle file.
    the pickle file is under the format:
    [folder name]/[image_name]/
                            k
                            uv_coord
                            jointmaps
                            heatmaps
    :param src: dataset folder
    :param dst: dst folder for new cropped dataset
    :param size: new image size (size x size)
    :return: None
    """
    train_dst = os.path.join(dst, 'train')
    test_dst = os.path.join(dst, 'test')
    label_paths = [os.path.join(src, 'labels', i) for i in os.listdir(os.path.join(src, 'labels'))]
    image_folders = [os.path.join(src, i) for i in os.listdir(src) if i != "labels" and i != "readme.txt"]

    image_paths = {}
    for folder in image_folders:
        images = os.listdir(folder)
        image_paths[os.path.basename(folder)] = [os.path.join(folder, i) for i in images]
    if DEBUG:
        print("image folders are : {}".format(image_paths.keys()))

    # for each image assign its xyz coordinate
    args = []

    train_labels = ["B1", "B2", "B3", "B5", "B6"]
    test_labels = ["B4"]

    for l_p in label_paths:
        folder = os.path.basename(l_p).split('_')[0]
        camera = os.path.basename(l_p).split('_')[-1][0:-4]

        images = image_paths[folder]
        labels = get_xyz_coord(l_p)
        images = list(filter(lambda x: os.path.basename(x).split("_")[0] == camera, images))
        if DEBUG:
            print(l_p, camera)
        for i in images:
            index = int(os.path.basename(i).split('_')[-1][0:-4])
            if os.path.basename(l_p)[0:2] in train_labels:
                destination = os.path.join(train_dst, folder, os.path.basename(i))
            elif os.path.basename(l_p)[0:2] in test_labels:
                destination = os.path.join(test_dst, folder, os.path.basename(i))
            else:
                raise ValueError
            args.append([i, destination, reorder(labels[index]), size])

    p = Pool()
    results = list(tqdm.tqdm(p.imap(image_process, args), ascii=True, total=len(args)))
    p.close()
    p.join()

    annotations_train = edict()
    annotations_test = edict()
    for r in results:
        destination, joints_map_path, heat_maps_path, anno_xyz, uv_coord, depth, k = r
        folder = os.path.basename(os.path.dirname(destination))
        image = os.path.basename(destination)

        if folder[0:2] in train_labels:
            annotations = annotations_train
        elif folder[0:2] in test_labels:
            annotations = annotations_test
        else:
            raise ValueError

        if folder not in annotations:
            annotations[folder] = edict()
            annotations[folder][image] = edict()
        else:
            annotations[folder][image] = edict()
        annotations[folder][image].xyz = anno_xyz
        annotations[folder][image].uv_coord = uv_coord
        annotations[folder][image].k = k
        annotations[folder][image].depth = depth
        annotations[folder][image].jointsmap_path = joints_map_path
        annotations[folder][image].heatmaps_path = heat_maps_path

    with open(os.path.join(train_dst, "stb_annotation_train.pickle"), "wb") as handle:
        pickle.dump(annotations_train, handle)

    with open(os.path.join(test_dst, "stb_annotation_test.pickle"), "wb") as handle:
        pickle.dump(annotations_test, handle)


if __name__ == "__main__":
    """ 
        STB stores its label under the following format 
    
        *_SK -> Intel Sense cameara
        *_BK -> bumble bee camera
    
        labels are stored in "handPara" and are in 3 X 21 X N
        3 are x, y, z
        21 are the joints 
        N are the total samples typically 1500
    
    #     note that only SK or Intel Sense camera contains  RBG, D and xyz dataset.
    # """
    #home_dir = 'datasets'
    home_dir = '.'
    src = os.path.join(home_dir, 'STB')
    dst = os.path.join(home_dir, 'STB_cropped')
    size = 256
    folders = ['train', 'test']

    if not os.path.exists(dst):
        os.makedirs(dst)
        for folder in folders:
            os.makedirs(os.path.join(dst, folder))
            os.makedirs(os.path.join(dst, folder, "B1Counting"))
            os.makedirs(os.path.join(dst, folder, "B1Random"))
            os.makedirs(os.path.join(dst, folder, "B2Counting"))
            os.makedirs(os.path.join(dst, folder, "B2Random"))
            os.makedirs(os.path.join(dst, folder, "B3Counting"))
            os.makedirs(os.path.join(dst, folder, "B3Random"))
            os.makedirs(os.path.join(dst, folder, "B4Counting"))
            os.makedirs(os.path.join(dst, folder, "B4Random"))
            os.makedirs(os.path.join(dst, folder, "B5Counting"))
            os.makedirs(os.path.join(dst, folder, "B5Random"))
            os.makedirs(os.path.join(dst, folder, "B6Counting"))
            os.makedirs(os.path.join(dst, folder, "B6Random"))
    main(src, dst, size)

