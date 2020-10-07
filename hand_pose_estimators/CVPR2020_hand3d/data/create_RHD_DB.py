#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
    Script to convert Rendered Handpose Dataset into binary files,
    which allows for much faster reading than plain image files.

    Set "path_to_db" and "set" accordingly.

    In order to use this file you need to download and unzip the dataset first.
"""
from __future__ import print_function, unicode_literals

import json
import pickle
import os
import cv2
import numpy as np
import scipy.misc
import struct
from multiprocessing import Pool

### No more changes below this line ###


# function to write the binary file
import tqdm as tqdm

FHD_BBOX = [0, 224, 0, 224]


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
        cv2.circle(image, (int(x), int(y)), 1, 255, 2)
        cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    if bbox is not None:
        cv2.rectangle(image, (bbox[0], bbox[3]), (bbox[1], bbox[2]), 255, 2)
    return image


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


def get_uv_coord(xyz, k_matrix, size):
    """get uv coordinate from xyz given k matrix"""
    null_array = np.array([[0.0, 0.0, 0.0]])
    uv_coord, _ = cv2.projectPoints(xyz, null_array, null_array, k_matrix, None)
    uv_coord, k_matrix = scale(np.reshape(uv_coord, (21, 2)).astype(np.float), k_matrix, FHD_BBOX, size)
    return uv_coord, k_matrix


def get_uv_wrapper(arg):
    xyz, k, size = arg
    return get_uv_coord(np.array(xyz), np.array(k), (size, size))


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


def main(src, dst, size):
    """
    create training and evaluating dataset for hand3d network given FHD src and dst paths
    :param src: path to data
    :param dst: output directory
    :param size: new size
    :return:
    """
    train_path = os.path.join(src, "training")
    # eval_path = os.path.join(src, "evaluation")

    dst_train = os.path.join(dst, 'training' )
    # dst_eval = os.path.join(dst, "evaluation")


    k_train = json.load(open(os.path.join(src, "training_K.json"), "r"))
    # k_test = np.array(json.load(open(os.path.join(src, "evaluation_K.json"), "r")))

    xyz_train = json.load(open(os.path.join(src, "training_xyz.json"), "r"))
    # xyz_test = np.array(json.load(open(os.path.join(src, "evaluation_K.json"), "r")))

    train_rgb_images = [os.path.join(train_path, "rgb", i) for i in os.listdir(os.path.join(train_path, "rgb"))]
    train_mask_images = [os.path.join(train_path, "mask", i) for i in os.listdir(os.path.join(train_path, "mask"))]

    if not os.path.isdir(train_path):
        raise OSError

    if not os.path.isdir(dst):
        os.mkdir(dst)
    if not os.path.isdir(dst_train):
        os.mkdir(dst_train)

    uv_args = []
    length = len(k_train)
    for i in range(len(train_rgb_images)):
        uv_args.append([xyz_train[i%length], k_train[i%length], size])

    p = Pool()
    results = list(p.map(get_uv_wrapper, uv_args))
    p.close()
    p.join()

    uv_coordinates = [i for i,_ in results]
    k_train = [j for _, j in results]

    print("compete getting uv coordinates: {} coordinates".format(len(uv_coordinates)))

    with open(os.path.join(dst_train, "training.bin"), 'wb') as file_handle:
        for i in range(len(train_rgb_images)):
            rgb_image = cv2.cvtColor(cv2.imread(train_rgb_images[i]), cv2.COLOR_BGR2RGB)
            mask_image = cv2.imread(train_mask_images[i % length], cv2.IMREAD_GRAYSCALE)

            rgb_image = cv2.resize(rgb_image, (size, size))
            mask_image = cv2.resize(mask_image, (size, size))

            uv_coord = uv_coordinates[i]
            xyz_coord = np.array(xyz_train[i % length])
            k = np.array(k_train[i % length])
            write_to_binary(file_handle, rgb_image, mask_image, xyz_coord, uv_coord, np.array([True] * uv_coord.shape[0]), k)

if __name__ == "__main__":
    import sys
    # arg[1] = data src
    # arg[2] = dst folder
    # arg[3] = resize
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))