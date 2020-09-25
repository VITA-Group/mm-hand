""" Basic example showing samples from the dataset.

    Chose the set you want to see and run the script:
    > python view_samples.py

    Close the figure to see the next sample.
"""
# from __future__ import print_function, unicode_literals
import cv2
import os
import pickle
from multiprocessing.pool import Pool

import numpy as np
import scipy.misc
import tqdm as tqdm
from easydict import EasyDict as edict

# chose between training and evaluation set
set = 'training'


# set = 'evaluation'


# auxiliary function
def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2 ** 8 + bottom_bits).astype('float32')
    depth_map /= float(2 ** 16 - 1)
    # depth_map *= 5.0
    return depth_map


def get_bbox(uv_coor, shape):
    xmin = ymin = 99999
    xmax = ymax = 0
    for [x, y] in uv_coor:
        xmin = min(xmin, int(x))
        xmax = max(xmax, int(x))
        ymin = min(ymin, int(y))
        ymax = max(ymax, int(y))
    xmin = max(0, xmin - 20)
    ymin = max(0, ymin - 20)

    xmax = min(shape[1], xmax + 20)
    ymax = min(shape[0], ymax + 20)

    return xmin, xmax, ymin, ymax


def getCameraMatrix():
    Fx = 614.878
    Fy = 615.479
    Cx = 313.219
    Cy = 231.288
    cameraMatrix = np.array([[Fx, 0, Cx],
                             [0, Fy, Cy],
                             [0, 0, 1]])
    return cameraMatrix


def plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color_fixed, linewidth=linewidth)

    axis.view_init(azim=-90., elev=90.)


def plot_hand(canvas, coords_hw, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    # Color Order:
    # thumb:blue
    # index: Cyan
    # middle: chartreuse
    # ring: orange
    # little: red
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 1), colors[0, :]),
             ((1, 2), colors[1, :]),
             ((2, 3), colors[2, :]),
             ((3, 4), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]

    axis.imshow(canvas)
    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], color_fixed, linewidth=linewidth)


# load annotations of this set
# with open(os.path.join(set, 'anno_%s.pickle' % set), 'rb') as fi:
# 	anno_all = pickle.load(fi)


order = [0, 4, 3, 2 ,1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18,17 ]


def image_process(src, dst, file_name, anno, size):
    '''
    function crops an image around the hand of a subject for RHD dataset. returns the new cropped anno dataset
    :param root_path: root dir containing all dataset
    :param anno: annotated dictionary found in anno_training.pickle
    :return: cropped a list of [cropped annotation, filename]
    '''

    # image root folders
    paths = {}
    names = ['color', 'depth', 'mask']
    # for n in names:
    #     paths['root_{}'.format(n)] = os.path.join(root_path, n)
    #     paths['dst_{}'.format(n)] = os.path.join(root_path, "cropped", n)

    # get K camera matrix
    matrix = np.array(anno['K'])


    if anno['xyz'].shape[0] > 21:  # two hands in frame
        left_anno = anno.copy()
        right_anno = anno.copy()
        left_anno['xyz'] = left_anno['xyz'][0:21]
        left_anno['uv_vis'] = left_anno['uv_vis'][:21, :]
        right_anno['xyz'] = right_anno['xyz'][21::, :]
        right_anno['uv_vis'] = right_anno['uv_vis'][21::, :]

        r1 = r2 = []

        r1 = image_process(src, dst, file_name + "_l", left_anno, size)

        r2 = image_process(src, dst, file_name + "_r", right_anno, size)

        return r1 + r2

        # if for some reason it has less than 21 points
    anno['xyz'] = anno['xyz'][:21, :][order]
    coor = anno['uv_vis'][:21, :][order]

    if sum(coor[:, -1]) != 21:
        return [None]
    # coor[:, :2] /= coor[:, 2:]
    # get cropped image of size (x y)
    xmin, xmax, ymin, ymax = get_bbox(coor[:, :2], (320, 320))

    if xmin > xmax or ymin > ymax:
        # print("Error at {}".format(file_name))
        return [None]
    # print (xmin, xmax, ymin, ymax)
    # scaling x, y and z coordinate to new coordinate
    coor[:, 0] = (coor[:, 0] - xmin) / (xmax - xmin + 1.) * size
    coor[:, 1] = (coor[:, 1] - ymin) / (ymax - ymin + 1.) * size

    shift = [[1, 0, -xmin],
             [0, 1, -ymin],
             [0, 0, 1]]

    xscale = 256. / (xmax - xmin + 1.)
    yscale = 256. / (ymax - ymin + 1.)

    scale = [[xscale, 0, 0],
             [0, yscale, 0],
             [0, 0, 1]]
    shift = np.array(shift)
    scale = np.array(scale)
    # modify kmap
    matrix = np.matmul(scale, np.matmul(shift, matrix))

    for n in names:
        img_name = file_name.split('_')[0]
        zero_padding = "0" * (5 - len(img_name))
        img_name = zero_padding + img_name + ".png"
        save_img_name = zero_padding + file_name.split('_')[0] + "_" + file_name[-1] + ".png"
        img_path = os.path.join(dst, n, save_img_name)

        img = cv2.imread(os.path.join(src, n, img_name))
        # img_shape = img.shape
        if img.shape[-1] == 3:
            img = img[ymin: ymax + 1, xmin:xmax + 1, :]
        else:
            img = img[ymin: ymax + 1, xmin:xmax + 1]

        try:
            img = cv2.resize(img, (size, size))
            if file_name.split('_')[-1] == 'r':
                "if file is right hand, we flip image"
                img = cv2.flip(img, 1)
            cv2.imwrite(img_path, img)
        except:
            return [None]
        # image = image[ymin:ymax + 1, xmin:xmax + 1]  # crop the image

    if file_name.split('_')[-1] == 'r':
        "if file is right hand, we flip image"
        coor[:, 0] = coor[:, 0] + 2 * (size / 2 - coor[:, 0])
    cropped_anno = dict()
    cropped_anno['K'] = matrix
    cropped_anno['uv_coord'] = coor[:, :2]
    cropped_anno['xyz'] = anno['xyz']
    cropped_anno['depth'] = anno['xyz'][:, -1]
    return [[file_name, cropped_anno]]


def gao_wrapper(args):
    src, dst, f, a, size = args
    return image_process(src, dst, f, a, size)


def main(src, dst, size):
    # path = os.path.join(src, 'training')
    # anno_name = "anno_training.pickle"
    path = os.path.join(src, 'evaluation')
    anno_name = "anno_evaluation.pickle"
    with open(os.path.join(path, anno_name), "rb") as f:
        anno = pickle.load(f)
    args = []
    for i in range(0, len(anno)):
        args.append([path, dst, '{}'.format(i), anno[i], size])
    p = Pool()
    results = list(tqdm.tqdm(p.imap(gao_wrapper, args), ascii=True, total=len(anno)))
    p.close()
    p.join()
    # flatten a list of list of list into a list of list, and get rid of any None type
    flatten = lambda l: [item for sublist in l for item in sublist if item is not None]
    results = flatten(results)
    cropped_annos = edict()
    # rename image to map index in cropped annos
    # 0_l = 0, 0_r = 1, etc ...
    i = 0
    while i < len(results):
        r = results[i]
        file_name, anno = r
        zero_padding = "0" * (5 - len(file_name.split('_')[0]))
        save_img_name = zero_padding + file_name.split('_')[0] + "_" + file_name[-1] + ".png"

        zero_padding = "0" * (5 - len("{}".format(i)))
        new_img_name = zero_padding + "{}".format(i) + ".png"

        # check for false detection that is the mask contains no hand mask at all
        mask = cv2.imread(os.path.join(dst, 'mask', save_img_name))
        folders = ['color', 'depth', 'mask']
        if np.max(mask) == 0 or np.max(mask) == 1:
            for f in folders:
                # print("false positive detected : {}".format(save_img_name))
                os.remove(os.path.join(dst, f, save_img_name))
            results.pop(i)
        else:
            for f in folders:
                os.rename(os.path.join(dst, f, save_img_name),
                          os.path.join(dst, f, new_img_name))
                if f not in cropped_annos:
                    cropped_annos[f] = edict()
                cropped_annos[f][new_img_name] = edict()
                for k, v in anno.items():
                    cropped_annos[f][new_img_name][k] = v
            i += 1
    with open(os.path.join(dst, "annotation.pickle"), 'wb') as handle:
        pickle.dump(cropped_annos, handle)


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


import sys

if __name__ == "__main__":
    """
    cropping hand for RHD dataset
    argv[1] = src
    argv[2] = dst
    argv[3] = new image size
    """
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])
        os.mkdir(os.path.join(sys.argv[2], 'color'))
        os.mkdir(os.path.join(sys.argv[2], 'depth'))
        os.mkdir(os.path.join(sys.argv[2], 'mask'))

    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))

    # path = sys.argv[1]
    # image = cv2.imread(os.path.join(path, "color", "00019.png"))
    # image = cv2.flip(image, 1)
    #
    # anno = pickle.load(open(os.path.join(path, "anno_evaluation.pickle"), 'rb'))
    #
    # anno19 = anno[19]
    # uv = anno19['uv_vis']
    # uv[:, 0] = uv[:, 0] + 2 * (image.shape[0] / 2 - uv[:, 0])
    #
    # image = draw(image, uv[:,:-1])
    # cv2.imwrite("test.png", image)
