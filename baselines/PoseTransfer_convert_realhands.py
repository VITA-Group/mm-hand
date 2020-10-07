
# coding: utf-8

# In[3]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import math
import h5py
import os
import sys
import io
from PIL import Image

from skimage.morphology import square, dilation, erosion
import scipy

#np.set_printoptions(threshold=sys.maxsize)
POSDIM = 25
FACEDIM = 70
HANDDIM = 21

VIZ_FACTOR = 1
PALM_COLOR = [1*VIZ_FACTOR]*3

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

H_scale = 1920//1
W_scale = 1080//1
H_offset = 960//1
W_offset = 540//1
BUFF = 50
TARGET_SIZE = 256


# In[4]:


def plot_hand_from_hand3d(coords_hw, canvas):
    """ Plots a hand stick figure into a matplotlib figure. """
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
    palm_area = [((0, 1), []),
                 ((1, 5), []),
                 ((5, 9), []),
                 ((9, 13), []),
                 ((13, 17), []),
                 ((17, 0), []),]
    # define connections and colors of the bones
    #print(coords_hw[-1]) # this is center ( the 22nd point)
    palm = []
    for connection, _ in palm_area:
        coord1 = coords_hw[connection[0]]
        palm.append([coord1[0],coord1[1]])
    #print(palm)
    cv2.fillConvexPoly(canvas,np.array( [palm], dtype=np.int32 ),PALM_COLOR)
    plot_xyz = []
    for connection, color in bones:
        coord1 = coords_hw[connection[0]]
        coord2 = coords_hw[connection[1]]
        coords = np.stack([coord1, coord2])
        # 0.5, 0.5 is the center
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        plot_xyz.append((x,y,z,color))

    plot_xyz.sort(key = lambda x: -x[2].mean())

    for x,y,z,color in plot_xyz:
        mX = x.mean()
        mY = y.mean()
        length = ((x[0]-x[1])**2 + (y[0]-y[1])**2) ** 0.5
        angle = math.degrees(math.atan2(y[0]-y[1], x[0]-x[1]))
        polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), 6), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [c* VIZ_FACTOR for c in color])

def hand3d2openpose(np_arr):
    #change keypoint ordering:
    np_arr[1:5,:] = np_arr[1:5,:][::-1,:]
    np_arr[5:9,:] = np_arr[5:9,:][::-1,:]
    np_arr[9:13,:] = np_arr[9:13,:][::-1,:]
    np_arr[13:17,:] = np_arr[13:17,:][::-1,:]
    np_arr[17:21,:] = np_arr[17:21,:][::-1,:]


# In[5]:


def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    values = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            distance = np.sqrt(float(i**2+j**2))
            if r+i>=0 and r+i<height and c+j>=0 and c+j<width:
                if 'Solid'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    values.append(1)
                elif 'Gaussian'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    if 4==var:
                        values.append( Gaussian_0_4.pdf(distance) * Ratio_0_4  )
                    else:
                        assert 'Only define Ratio_0_4  Gaussian_0_4 ...'
    return indices, values

def _sparse2dense(indices, values, shape):
    dense = np.zeros(shape[:2])
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        #k = indices[i][2]
        #dense[r,c,k] = values[i]
        dense[r,c] = values[i]
    return dense

def _visualizePose(pose):
    # pdb.set_trace()
    if 3==len(pose.shape):
        pose = pose.max(axis=-1, keepdims=True)
        pose = np.tile(pose, (1,1,3))
    elif 2==len(pose.shape):
        pose = np.expand_dims(pose, -1)
        pose = np.tile(pose, (1,1,3))
    return pose * 255

def _getPoseMask(peaks, height, width, radius=8, var=4, mode='Solid'):
    boneSeq = [[1,2], [2,3], [3,4], 
               [5,6], [6,7], [7,8], \
               [9,10], [10,11], [11,12], \
               [13,14], [14,15], [15,16], \
               [17,18], [18,19], [19,20]] #
    palmSeq = [[0,1], [1,5], [5,9], [9,13], [13,17], [17,0]]
    indices = []
    values = []

    for bone in boneSeq:
        p0 = peaks[bone[0]]
        p1 = peaks[bone[1]]
        #print(p0)
        #print(p1)
        if len(p0) != 0 and len(p1) != 0:
            c0 = p0[0]
            r0 = p0[1]
            
            c1 = p1[0]
            r1 = p1[1]
            
            ind, val = _getSparseKeypoint(r0, c0, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
            ind, val = _getSparseKeypoint(r1, c1, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)

            distance = np.sqrt((r0-r1)**2 + (c0-c1)**2)
            sampleN = int(distance/radius)
            if sampleN>1:
                for i in range(1,sampleN):
                    r = r0 + (r1-r0)*i/sampleN
                    c = c0 + (c1-c0)*i/sampleN
                    ind, val = _getSparseKeypoint(r, c, 0, height, width, radius, var, mode)
                    indices.extend(ind)
                    values.extend(val)
    palm = []
    for connection in palmSeq:
        p0 = peaks[connection[0]]
        palm.append([p0[0], p0[1]])
    #print(palm)
    canvas = np.zeros((height, width, 1), np.uint8)

    cv2.fillConvexPoly(canvas,np.array([palm], dtype=np.int32 ),1)
    c_arr, r_arr, _ = np.nonzero(canvas)
    print(c_arr.shape)
    print(r_arr.shape)
    for i in range(c_arr.shape[0]):
        indices.append([c_arr[i], r_arr[i], 0])
        values.append(1)
    shape = [height, width, 1]
    ## Fill body
    dense = np.squeeze(_sparse2dense(indices, values, shape))
    dense = dilation(dense, square(5))
    dense = erosion(dense, square(5))
    return dense

def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]
        if 0!=len(p):
            r = p[1]
            c = p[0]
            ind, val = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
    shape = [height, width, channel]
    return indices, values, shape


# In[6]:


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
        
        if x_max - x_min > y_max - y_min:
            delta = int((x_max - x_min - (y_max - y_min)) / 2)
            y_max = min(H_scale-1,y_max+delta)
            y_min = max(1,y_min-delta)
        else:
            delta = int((y_max - y_min - (x_max - x_min)) / 2)
            x_max = min(W_scale-1,x_max+delta)
            x_min = max(1,x_min-delta)
        bounding_box = [x_min, y_min, x_max, y_max]
        return bounding_box
    else:
        return None


# In[7]:


def draw(coords, index):
    path_img_out1 = "fixed_view/sparse_pose/" + str(index).zfill(4) + ".png"
    path_img_out2 = "fixed_view/pose_mask/" + str(index).zfill(4) + ".png"
        
    indices_r4, values_r4, shape = _getSparsePose(coords, TARGET_SIZE, TARGET_SIZE, 22, radius=4, mode='Solid')
    pose_dense_r4 = _sparse2dense(indices_r4, values_r4, shape)
    print(pose_dense_r4.shape)
    cv2.imwrite(path_img_out1, _visualizePose(pose_dense_r4))

    pose_mask = _getPoseMask(coords, TARGET_SIZE, TARGET_SIZE, radius=8, var=4, mode='Solid')
    print(pose_mask.shape)
    cv2.imwrite(path_img_out2, _visualizePose(pose_mask))

def coords_transform(coords, bounding_box):
    h_ratio, w_ratio = TARGET_SIZE / (bounding_box[3] - bounding_box[1]), TARGET_SIZE / (
                    bounding_box[2] - bounding_box[0])
    coords[:, 0], coords[:, 1] = (coords[:, 0] - bounding_box[0]) * w_ratio, (
                    coords[:, 1] - bounding_box[1]) * h_ratio
    return coords


# In[8]:


offset = 0
coords_dict = {}

ANNOTATION_FORMAT = "Real_Hands/real_hands_ims_annos/20190716_pix%s_%s_reid_json.h5"
IMAGE_PATH_FORMAT = "Real_Hands/real_hands_ims_annos/20190716_pix%s_%s_jpg.h5"
data_pairs = [('001','1.fingertapping_hand_left'),('001','2.fist_hand_left'),
              ('002','1.fingertapping_hand_left'),('002','2.fist_hand_left'),
              ('003','1.fingertapping_hand_left'),('003','2.fist_hand_left')]
for hand_id,gesture in data_pairs:
    annos = h5py.File(ANNOTATION_FORMAT % (hand_id, gesture), 'r')
    im_paths = h5py.File(IMAGE_PATH_FORMAT % (hand_id, gesture), 'r')["binary_jpg"]
    keypoints = annos['keypoint'][:,POSDIM+FACEDIM:POSDIM+FACEDIM+HANDDIM,:2]
    print(keypoints.shape)
    hand3d_file_path = "Real_Hands/real_hands_ims_annos/hand3d/pix%s/%s"%(hand_id, gesture)
    nframes = keypoints.shape[0]
    for i in range(nframes):
        coords = np.load("%s/%04d.npy"%(hand3d_file_path,i))
        hand3d2openpose(coords)
        coords[:,:2] = keypoints[i,:,::-1]
        fig2 = np.zeros((H_scale//3, W_scale//3, 3), np.uint8)
        coords[:,0] = W_scale//3 - coords[:,0]
        plot_hand_from_hand3d(coords, fig2)
        gray = cv2.cvtColor(fig2, cv2.COLOR_RGB2GRAY)
        bounding_box = get_bbox(coords, W_scale, H_scale)
        coords = coords_transform(coords, bounding_box)
        #print(coords.shape)
        #print(coords)
        coords_dict[os.path.join('Real_Hands/real_hands_cropped', str(i+offset).zfill(5) + '.png')] = coords
    offset += nframes


# In[15]:


print(coords_dict['Real_Hands/real_hands_cropped/00000.png'])


# In[10]:


import random
import csv

pairs = []
for i in range(10000):
    m = random.randint(5816, 12717)
    n = random.randint(5816, 12717)
    pairs.append((str(m).zfill(5) + '.png', str(n).zfill(5) + '.png'))
with open('realhands-pairs-train.csv',mode='w') as csv_file:
    fieldnames = ['from','to']
    writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames, escapechar=',', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for pair in pairs:
        writer.writerow({'from':pair[0], 'to':pair[1]})


# In[18]:


# i = 0
# for key, value in coords_dict.items():
#     print(i)
#     #print(len(keypoint_coord3d_v))
#     #print(keypoint_coord3d_v)
#     print(key)
#     draw(value, i)
#     i += 1
    
import csv
with open('realhands-annotation-test.csv',mode='w') as csv_file:
    fieldnames = ['name','keypoints_y','keypoints_x']
    writer = csv.DictWriter(csv_file, delimiter=':', fieldnames=fieldnames, escapechar=':', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for key, value in coords_dict.items():
        #print(key)
        index = int(key.split('/')[-1].split('.')[0])
        #print(index)
        if index <= 5815:
            writer.writerow({'name':key.split('/')[-1], 
                             'keypoints_y':list(np.int_(value[:,1])),
                             'keypoints_x':list(np.int_(value[:,0]))})
            print('{}:{};{}'.format(key.split('/')[-1], list(np.int_(value[:,1])), list(np.int_(value[:,0]))))
            print(value)


# In[ ]:


with open('realhands-annotation-train.csv',mode='w') as csv_file:
    fieldnames = ['name','keypoints_y','keypoints_x']
    writer = csv.DictWriter(csv_file, delimiter=':', fieldnames=fieldnames, escapechar=':', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for key, value in coords_dict.items():
        print(key)
        index = int(key.split('/')[-1].split('.')[0])
        print(index)
        if index > 5815:
            writer.writerow({'name':key.split('/')[-1], 
                             'keypoints_y':list(np.int_(value[:,1])),
                             'keypoints_x':list(np.int_(value[:,0]))})

