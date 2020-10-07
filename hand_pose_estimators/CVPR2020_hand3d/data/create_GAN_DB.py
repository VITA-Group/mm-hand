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
from multiprocessing import Pool

K_intrinsic = np.array([[617.173, 0, 315.453],
                        [0, 617.173, 242.259],
                        [0, 0, 1]])


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
        cv2.circle(image, (int(x), int(y)), 2, 255, 2)
        cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    if bbox is not None:
        cv2.rectangle(image, (bbox[0], bbox[3]), (bbox[1], bbox[2]), 255, 2)
    return image

def add_padding(coords):
    """ add an additional 21 kp entries to coordinate to conform with hand3d create_binary_db format """
    padding = np.zeros(coords.shape)
    return np.concatenate([coords, padding], axis=0)

def get_uv(f_path):
    with open(f_path, 'r') as f:
        for line in f.readlines():
            uv = np.array([float(i) for i in line.split(',')], dtype=np.float32)
            uv = uv.reshape((21, 2))
    uv = add_padding(uv)
    return uv

def get_xyz(f_path):
    with open(f_path, 'r') as f:
        for line in f.readlines():
            xyz = np.array([float(i) for i in line.split(',')], dtype=np.float32)
            xyz = xyz.reshape((21, 3))
    xyz = add_padding(xyz)
    return xyz

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

def multitask_handler(args):
    src, target, dst = args
    save_path = os.path.join(dst, "{}_{}.bin".format(os.path.basename(src), target))
    with open(save_path, 'wb') as handler:
        subject = target
        uv_f = subject + "_" + "joint2D.txt"
        xyz_f = "{}_joint_pos_global.txt".format(subject)
        image_f = "{}_color_composed.png".format(subject)
        uv_coord = get_uv(os.path.join(src, uv_f))
        xyz = get_xyz(os.path.join(src, xyz_f))
        image = cv2.imread(os.path.join(src, image_f))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.ones(image.shape[:-1], dtype=np.int32)
        viz = np.array([1 for _ in range(21)])
        assert(image.shape == (256, 256,3))
        write_to_binary(handler, image, mask, xyz, uv_coord, viz, K_intrinsic)
    return save_path


def multi_task_write(dst, args):
    p = Pool()
    results = list(tqdm.tqdm(p.imap(multitask_handler, args), total=len(args), desc= "generating binaries"))
    p.close()
    p.join()

    with open(dst, 'wb') as handler:
        for path in tqdm.tqdm(results, desc="combining binaries"):
            with open(path, 'rb') as f:
                handler.write(f.read())
            os.remove(path)


def main(src, dst):
    iter  = os.walk(src, True)
    _, dirs, _ = next(iter)

    train_subject = dirs[0: int(.8*len(dirs))]
    eval_subjects = dirs[int(.8*len(dirs))::]

    def get_args(iterator):
        args= []
        for t in iterator:
            root = os.path.join(src, t)
            files = [i for i in os.listdir(os.path.join(src, t))]
            files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
            for i in range(0, len(files), 5):
                subject = files[i].split('_')[0]
                args.append([root, subject, dst])
        return args

    train_args = get_args(train_subject)
    eval_args = get_args(eval_subjects)

    multi_task_write(os.path.join(dst, "gan_train.bin"), train_args)
    multi_task_write(os.path.join(dst, "gan_evaluate.bin"), eval_args)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
