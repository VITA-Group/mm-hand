
# coding: utf-8

# In[2]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import math
import h5py
import os
import sys
np.set_printoptions(threshold=sys.maxsize)

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
dic = json.load(open("Synthetic_Hands/synthetic_hands/fixed_view0/label_fixed_view0_2000.json","r"))
H_scale = 1920//1
W_scale = 1080//1
H_offset = 960//1
W_offset = 540//1
import torch

def plot_hand_cv2(coords_hw, canvas):
    """ Plots a hand stick figure into a matplotlib figure. """

    # define connections and colors of the bones
    #print(coords_hw[-1]) # this is center ( the 22nd point)
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

    ret_coor = []
    """

    """
    palm = []
    for connection, _ in [((0, 1), []),
                            ((1, 5), []),
                            ((5, 9), []),
                            ((9, 13), []),
                            ((13, 17), []),
                            ((17, 0), []),]:
        coord1 = coords_hw[connection[0]]
        palm.append([int((coord1[0]-.5)* W_scale+ W_offset ), int(-(coord1[1]- .5)* H_scale+ H_offset)])
    #print(palm)
    cv2.fillConvexPoly(canvas,np.array( [palm], dtype=np.int32 ),PALM_COLOR)
    for connection, color in bones:
        coord1 = coords_hw[connection[0]]
        coord2 = coords_hw[connection[1]]
        coords = np.stack([coord1, coord2])
        # 0.5, 0.5 is the center
        x = (coords[:, 0]-0.5)* W_scale+ W_offset
        y = -(coords[:, 1]-0.5)* H_scale+ H_offset
        mX = x.mean()
        mY = y.mean()
        length = ((x[0]-x[1])**2 + (y[0]-y[1])**2) ** 0.5
        angle = math.degrees(math.atan2(y[0]-y[1], x[0]-x[1]))
        polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), 16), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, color)


# In[23]:


class Colorize(object):
    def __init__(self, n):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def add_color(self, gray_image):
        size = gray_image.shape
        color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)
        print(self.cmap.shape)
        for label in range(self.cmap.shape[0]):
            mask = (gray_image == label)
            print(mask.shape)
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
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
    
    
def draw(keypoint_coord3d_v, index, azim_z, elev_e):
    fig2 = np.zeros((H_scale, W_scale, 3), np.uint8)
    plot_hand_cv2(keypoint_coord3d_v, fig2)

    path_img_out2 = "fixed_view/train_label/" + str(index).zfill(4) + ".png"

    gray = cv2.cvtColor(fig2, cv2.COLOR_RGB2GRAY)
    label_tensor = Colorize(17).add_color(gray)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_numpy.astype(np.uint8)
    cv2.imwrite(path_img_out2, label_numpy)


BUFF = 50
for i in range(len(dic))[0:]:
    keypoint_coord3d_v = dic[str(i).zfill(7)]['perspective']
    draw(keypoint_coord3d_v, i, 90, -90)


# In[25]:


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
            print('id:{}; str_id:{}'.format(id, str_id))
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
            print('r:{};g:{};b:{}'.format(r,g,b))
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

labelcolormap(21)

