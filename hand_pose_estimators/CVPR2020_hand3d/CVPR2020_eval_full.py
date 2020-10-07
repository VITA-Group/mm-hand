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
from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from easydict import EasyDict as edict
from data.STB_dataset import STBdataset
from data.RHD_dataset import RHDdataset
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import EvalUtil, detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

if __name__ == '__main__':
    # images to be shown
    opt = edict()
    opt.dataroot = "./datasets/stb-dataset/test"
    opt.isTrain = False
    data = STBdataset(opt)
    # opt.dataroot = "./datasets/rhd-dataset/test"
    # opt.isTrain = False
    # data = RHDdataset(opt)

    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, \
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference_crop(image_tf, hand_side_tf, evaluation)
    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)

    # Feed image list through network
    eval2d = EvalUtil()
    eval3d = EvalUtil()
    import tqdm
    import cv2

    for i in tqdm.tqdm(range(len(data))):
        sample = data[i]
        image_v = sample['image']

        hand_scoremap_v, image_crop_v, scale_v, center_v, \
        keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                             keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                            feed_dict={image_tf: image_v})

        hand_scoremap_v = np.squeeze(hand_scoremap_v)
        image_crop_v = np.squeeze(image_crop_v)
        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
        keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v) * sample['keypoint_scale']
        keypoint_xyz21_gt = np.squeeze(sample['keypoint_xyz21'])
        keypoint_xyz21_gt -= keypoint_xyz21_gt[0]
        keypoint_uv_gt = np.squeeze(sample['keypoint_uv21'])
        # post processing
        image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
        coord_hw_pred_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
        coord_uv_pred_crop = np.stack([coord_hw_pred_crop[:, 1], coord_hw_pred_crop[:, 0]], 1)

        kp_vis2d = np.ones_like(keypoint_uv_gt[:, 0])
        kp_vis = np.ones_like(keypoint_xyz21_gt[:, 0])
        eval2d.feed(keypoint_uv_gt, kp_vis2d, coord_uv_pred_crop)
        eval3d.feed(keypoint_xyz21_gt, kp_vis, keypoint_coord3d_v)

        # for i, kp in enumerate(keypoint_uv_gt):
            # kp = tuple(kp.astype(np.uint8))
            # image_crop_v = cv2.circle(image_crop_v, kp, 2, [255,255,255], 2)
            # image_crop_v = cv2.putText(image_crop_v, str(i),
                                        # kp,cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        # [0,255,0])

        # import pdb; pdb.set_trace()
        # cv2.imwrite("test.png", image_crop_v)
mean, median, auc, _, _ = eval3d.get_measures(0.0, 0.050, 20)
print('Evaluation results for 3d ')
print('Average mean EPE: %.3f mm' % (mean * 1000))
print('Average median EPE: %.3f mm' % (median * 1000))
print('Area under curve: %.3f' % auc)

mean, median, auc, _, _ = eval2d.get_measures(0.0, 30.0, 20)
print('Evaluation results 2d :')
print('Average mean EPE: %.3f pixels' % mean)
print('Average median EPE: %.3f pixels' % median)
print('Area under curve: %.3f' % auc)
