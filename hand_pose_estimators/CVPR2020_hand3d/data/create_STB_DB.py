import pickle
from multiprocessing.pool import Pool

import numpy as np
import tqdm as tqdm
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os
import cv2
from easydict import EasyDict as edict
import numpy as np
import sys
import struct

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


PALM_COLOR = [10]*3
THUMB_COLOR1 = [20]*3
THUMB_COLOR2 = [30]*3
THUMB_COLOR3 = [40]*3
INDEX_COLOR1 = [50]*3
INDEX_COLOR2 = [60]*3
INDEX_COLOR3 = [70]*3
MIDDLE_COLOR1 = [80]*3
MIDDLE_COLOR2 = [90]*3
MIDDLE_COLOR3 = [100]*3
RING_COLOR1 = [110]*3
RING_COLOR2 = [120]*3
RING_COLOR3 = [130]*3
PINKY_COLOR1 = [140]*3
PINKY_COLOR2 = [150]*3
PINKY_COLOR3 = [160]*3

#
# ordering: palm center(not wrist or hand center), little_mcp, little_pip, little_dip, little_tip, ring_mcp, ring_pip,
# ring_dip, ring_tip, middle_mcp, middle_pip, middle_dip, middle_tip, index_mcp, index_pip, index_dip, index_tip,
# thumb_mcp, thumb_pip, thumb_dip, thumb_tip.

# remapping labels to fit with standard labeling.
STB_TO_STD = [0, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4]

def create_jointsmap(uv_coord, size):
    """ Plots a hand stick figure into a matplotlib figure. """

    # define connections and colors of the bones
    #print(coords_hw[-1]) # this is center ( the 22nd point)
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
                            ((17, 0), []),]:
        coord1 = uv_coord[connection[0]]
        palm.append([int(coord1[0]), int(coord1[1])])
        # palm.append([int((coord1[0]-.5)* W_scale+ W_offset ), int(-(coord1[1]- .5)* H_scale+ H_offset)])
    #print(palm)
    cv2.fillConvexPoly(canvas,np.array([palm], dtype=np.int32 ), PALM_COLOR)
    for connection, color in bones:
        coord1 = uv_coord[connection[0]]
        coord2 = uv_coord[connection[1]]
        coords = np.stack([coord1, coord2])
        # 0.5, 0.5 is the center
        x = coords[:, 0]
        y = coords[:, 1]
        mX = x.mean()
        mY = y.mean()
        length = ((x[0]-x[1])**2 + (y[0]-y[1])**2) ** 0.5
        angle = np.math.degrees(np.math.atan2(y[0]-y[1], x[0]-x[1]))
        polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), 16), int(angle), 0, 360, 1)
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


def image_process(arg):
    img_path, anno_xyz, size = arg
    image = cv2.imread(img_path)
    camera, mode, index = os.path.basename(img_path).split("_")
    uv_coor = get_uv_coord(mode, camera, anno_xyz)
    # bbox = get_bounding_box(uv_coor, image.shape)
    #
    # xmin, xmax, ymin, ymax = bbox
    #
    # image = image[ymin:ymax + 1, xmin:xmax + 1]  # crop the image
    # image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = np.ones((image.shape[:2]), dtype=np.int32)

    if camera == "BB":
        K = I_BB.K.copy()
    else:
        if mode == "color":
            K = I_SK.K_depth.copy()
        else:
            K = I_SK.K_color.copy()
    # uv_coor, k = scale(uv_coor, K, bbox, (size, size))

    return [image, anno_xyz, uv_coor, mask, K]


def get_images(args):
    for arg in args:
        yield image_process(arg)


def add_padding(coords):
    """ add an additional 21 kp entries to coordinate to conform with hand3d create_binary_db format """
    padding = np.zeros(coords.shape)
    return np.concatenate([coords, padding], axis=0)


def write_to_binary(file_handle, image, mask, kp_coord_xyz, kp_coord_uv, kp_visible, K_mat):
    """" Writes records to an open binary file. """
    bytes_written = 0
    # 1. write kp_coord_xyz
    for coord in kp_coord_xyz:
        file_handle.write(struct.pack('f', coord[0]))
        file_handle.write(struct.pack('f', coord[1]))
        file_handle.write(struct.pack('f', coord[2]))
    bytes_written += 4*kp_coord_xyz.shape[0]*kp_coord_xyz.shape[1]

    # 2. write kp_coord_uv
    for coord in kp_coord_uv:
        file_handle.write(struct.pack('f', coord[0]))
        file_handle.write(struct.pack('f', coord[1]))
    bytes_written += 4*kp_coord_uv.shape[0]*kp_coord_uv.shape[1]

    # 3. write camera intrinsic matrix
    for K_row in K_mat:
        for K_element in K_row:
            file_handle.write(struct.pack('f', K_element))
    bytes_written += 4*9

    file_handle.write(struct.pack('B', 255))
    file_handle.write(struct.pack('B', 255))
    bytes_written += 2

    # 4. write image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            file_handle.write(struct.pack('B', image[x, y, 0]))
            file_handle.write(struct.pack('B', image[x, y, 1]))
            file_handle.write(struct.pack('B', image[x, y, 2]))
    bytes_written += 4*image.shape[0]*image.shape[1]*image.shape[2]

    # 5. write mask
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            file_handle.write(struct.pack('B', mask[x, y]))
    bytes_written += 4*mask.shape[0]*mask.shape[1]

    # 6. write visibility
    for x in range(kp_visible.shape[0]):
        file_handle.write(struct.pack('B', kp_visible[x]))
    bytes_written += kp_visible.shape[0]


def main(src, dst, size):
    """
    :param src: dataset folder
    :param dst: where to save the bin.
    :param size: new image size (size x size)
    :return: None
    """

    label_paths = [os.path.join(src, 'labels', i) for i in os.listdir(os.path.join(src, 'labels'))]
    image_folders = [os.path.join(src, i) for i in os.listdir(src) if i != "labels" and i != "readme.txt"]

    image_paths = {}
    for folder in image_folders:
        images = os.listdir(folder)
        image_paths[os.path.basename(folder)] = [os.path.join(folder, i) for i in images]
    if DEBUG:
        print("image folders are : {}".format(image_paths.keys()))

    # for each image assign its xyz coordinate
    args_train = []
    args_test = []
    train_labels = ["B1", "B2", "B3", "B5", "B6"]
    test_labels = ["B4"]

    for l_p in label_paths:
        folder = os.path.basename(l_p).split('_')[0]
        camera = os.path.basename(l_p).split('_')[-1][0:-4]

        images = image_paths[folder]
        labels = get_xyz_coord(l_p)
        images = list(filter(lambda x: os.path.basename(x).split("_")[0] == camera, images))
        if camera == "SK":
            images = list(filter(lambda x: os.path.basename(x).split("_")[1] != "depth", images))
        else:
            images = list(filter(lambda x: os.path.basename(x).split("_")[1] != "left", images))
        if DEBUG:
            print(l_p, camera)
        for i in images:
            index = int(os.path.basename(i).split('_')[-1][0:-4])
            if os.path.basename(l_p)[0:2] in train_labels:
                args_train.append([i, labels[index], size])
            elif os.path.basename(l_p)[0:2] in test_labels:
                args_test.append([i, labels[index], size])
            else:
                raise ValueError

    if not os.path.exists(dst):
        os.makedirs(dst)
    with open(os.path.join(dst, "stb_training.bin"), 'wb') as file_handle:
        for r in tqdm.tqdm(get_images(args_train), total=len(args_train)):
            image, xyz, uv_coord, mask_image, k = r
            xyz = add_padding(xyz)
            uv_coord = add_padding(uv_coord)
            write_to_binary(file_handle, image, mask_image, xyz, uv_coord, np.array([True] * len(uv_coord)), k)

    with open(os.path.join(dst, "stb_evaluate.bin"), 'wb') as file_handle:
        for r in tqdm.tqdm(get_images(args_test), total=len(args_test)):
            image, xyz, uv_coord, mask_image, k = r
            xyz = add_padding(xyz)
            uv_coord = add_padding(uv_coord)
            write_to_binary(file_handle, image, mask_image, xyz, uv_coord, np.array([True] * len(uv_coord)), k)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))

