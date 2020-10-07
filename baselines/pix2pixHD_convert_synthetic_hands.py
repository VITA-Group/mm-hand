
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
def return_img_path(i):
    return "Synthetic_Hands/synthetic_hands/fixed_view0/%04d.png" % (i+1)

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

def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
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
    print(coords_hw[-1]) # this is center ( the 22nd point)
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
    ret_coor = []
    for connection, color in bones:
        coord1 = coords_hw[connection[0]]
        coord2 = coords_hw[connection[1]]
        coords = np.stack([coord1, coord2])
        # 0.5, 0.5 is the center
        x = (coords[:, 0]-0.5)* W_scale+ W_offset
        y = -(coords[:, 1]-0.5)* H_scale+ H_offset
        axis.plot( x , y , color=color, linewidth=linewidth)
        ret_coor.append( [x[1], y[1]] )
        if connection[1] == 1:
            ret_coor.append( [x[0], y[0]] )

    return np.asarray(ret_coor)


# In[4]:


BUFF = 50
TARGET_SIZE = 512
def crop_scale(img,bbox):
    x_min, y_min, x_max, y_max = bbox
    x_min = int(np.floor(max(1,x_min-BUFF)))
    y_min = int(np.floor(max(1,y_min-BUFF)))
    x_max = int(np.ceil(min(W_scale-1,x_max+BUFF)))
    y_max = int(np.ceil(min(H_scale-1,y_max+BUFF)))
    crop = img[y_min:y_max,x_min:x_max]
    return crop, (x_min, y_min, x_max, y_max)

def get_bbox(point_np,imw,imh):
    if point_np.shape[0] > 0:
        x_col = point_np[:,0]
        y_col = point_np[:,1]
        x_min = max(1,np.nanmin(x_col)-BUFF)
        y_min = max(1,np.nanmin(y_col)-BUFF)
        x_max = min(imw-1,np.nanmax(x_col)+BUFF)
        y_max = min(imh-1,np.nanmax(y_col)+BUFF)
        return [x_min, y_min, x_max, y_max]
    else:
        return None

def draw(keypoint_coord3d_v, index, azim_z, elev_e):
    fig2 = np.zeros((H_scale, W_scale, 3), np.uint8)

    plot_hand_cv2(keypoint_coord3d_v, fig2)

    path_img_out2 = "fixed_view/train_label/" + str(index).zfill(4) + ".png"
    path_img_out3 = "fixed_view/train_img/" + str(index).zfill(4) + ".png"



    gray = cv2.cvtColor(fig2, cv2.COLOR_RGB2GRAY)
    print(np.max(gray))
    #ret, gray = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)

    coords = np.array(keypoint_coord3d_v)[:,:2]
    coords[:, 0] = (coords[:, 0]-0.5)* W_scale+ W_offset
    coords[:, 1] = -(coords[:, 1]-0.5)* H_scale+ H_offset

    x_min, y_min, x_max, y_max = get_bbox(coords,W_scale,H_scale)
    if x_max - x_min > y_max - y_min:
        delta = int((x_max - x_min - (y_max - y_min)) / 2)
        y_max = min(H_scale-1,y_max+delta)
        y_min = max(1,y_min-delta)
    else:
        delta = int((y_max - y_min - (x_max - x_min)) / 2)
        x_max = min(W_scale-1,x_max+delta)
        x_min = max(1,x_min-delta)
    bounding_box = [x_min, y_min, x_max, y_max]
    binary_img_crop, _ = crop_scale(gray,bounding_box)
    orig_img_crop, _ = crop_scale(cv2.imread(return_img_path(index)),bounding_box)
    binary_img_crop = cv2.resize( binary_img_crop , (TARGET_SIZE, TARGET_SIZE) )

    orig_img_crop = cv2.resize( orig_img_crop , (TARGET_SIZE, TARGET_SIZE) )

    cv2.imwrite(path_img_out2, binary_img_crop)
    cv2.imwrite(path_img_out3, orig_img_crop)

BUFF = 50
for i in range(len(dic))[0:]:
    keypoint_coord3d_v = dic[str(i).zfill(7)]['perspective']
    draw(keypoint_coord3d_v, i, 90, -90)

