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


# auxiliary function
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



order = [0, 4, 3, 2 ,1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18,17 ]


def image_process(src, dst, file_name, anno, size):
    '''
    function crops an image around the hand of a subject for RHD dataset. returns the new cropped anno dataset
    :param root_path: root dir containing all dataset
    :param anno: annotated dictionary found in anno_training.pickle
    :return: cropped a list of [cropped annotation, filename]
    '''

    # image root folders
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


def main(src, dst, size):
    if not os.path.exists(dst):
        os.mkdir(dst)

    folders = ['training', 'evaluation']
    for folder in folders:
        src_path = os.path.join(src, folder)
        dst_path = os.path.join(dst, folder)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
            os.makedirs(os.path.join(dst_path, 'color'))
            os.makedirs(os.path.join(dst_path, 'depth'))
            os.makedirs(os.path.join(dst_path, 'mask'))

        anno_name = "anno_{}.pickle".format(folder)
        with open(os.path.join(src_path, anno_name), "rb") as f:
            anno = pickle.load(f)
        args = []
        for i in range(0, len(anno)):
            args.append([src_path, dst_path, '{}'.format(i), anno[i], size])
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
            mask = cv2.imread(os.path.join(dst_path, 'mask', save_img_name))
            subfolders = ['color', 'depth', 'mask']
            if np.max(mask) == 0 or np.max(mask) == 1:
                for f in subfolders:
                    # print("false positive detected : {}".format(save_img_name))
                    os.remove(os.path.join(dst_path, f, save_img_name))
                results.pop(i)
            else:
                for f in subfolders:
                    os.rename(os.path.join(dst_path, f, save_img_name),
                              os.path.join(dst_path, f, new_img_name))
                    if f not in cropped_annos:
                        cropped_annos[f] = edict()
                    cropped_annos[f][new_img_name] = edict()
                    for k, v in anno.items():
                        cropped_annos[f][new_img_name][k] = v
                i += 1
        with open(os.path.join(dst_path, "rhd_annotation_{}.pickle".format(folder)), 'wb') as handle:
            pickle.dump(cropped_annos, handle)


if __name__ == "__main__":
    """
    cropping hand for RHD dataset
    """
    # home_dir = 'datasets'
    home_dir = '.'
    src = os.path.join(home_dir, 'RHD')
    dst = os.path.join(home_dir, 'RHD_cropped')
    size = 256

    main(src, dst, size)

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
