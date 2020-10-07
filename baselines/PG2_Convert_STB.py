#!/usr/bin/env python
# coding: utf-8

# In[16]:


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

TARGET_SIZE = 256


# In[ ]:


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


# In[35]:


def draw(coords, name, path_img_out1, path_img_out2):
    path1 = os.path.join(path_img_out1, name.split('/')[0])
    if not os.path.exists(path1):
        os.makedirs(path1)
    path2 = os.path.join(path_img_out2, name.split('/')[0])
    if not os.path.exists(path2):
        os.makedirs(path2)
    indices_r4, values_r4, shape = _getSparsePose(coords, TARGET_SIZE, TARGET_SIZE, 22, radius=4, mode='Solid')
    pose_dense_r4 = _sparse2dense(indices_r4, values_r4, shape)
    print(pose_dense_r4.shape)
    cv2.imwrite(os.path.join(path_img_out1, name), _visualizePose(pose_dense_r4))

    pose_mask = _getPoseMask(coords, TARGET_SIZE, TARGET_SIZE, radius=8, var=4, mode='Solid')
    print(pose_mask.shape)
    cv2.imwrite(os.path.join(path_img_out2, name), _visualizePose(pose_mask))

path_img_out1 = 'PG2_STB/sparse_pose'
if not os.path.exists(path_img_out1):
    os.makedirs(path_img_out1)

path_img_out2 = 'PG2_STB/pose_mask'
if not os.path.exists(path_img_out2):
    os.makedirs(path_img_out2)
    
for name, coords in coords_dict_test.items():
    print(name)
    draw(coords, name, path_img_out1, path_img_out2)


# In[32]:


import pickle
annos = pickle.load(open("STB_cropped/train/annotation.pickle", "rb"))


# In[27]:


name_dict = {}
for folder_name in annos.keys():
    name_dict[folder_name] = []
    for img_name in annos[folder_name]:
        if 'SK_color' in img_name:
            name_dict[folder_name].append(img_name)
print(len(name_dict['B6Random']))

pairs_train = []
for folder_name in annos.keys():
    if folder_name in ['B6Counting', 'B6Random']:
        continue    
    nframes = len([image for image in list(annos[folder_name].keys()) if 'SK_color' in image])
    print(nframes)
    for i in range(nframes):
        m = random.randint(0, nframes-1)
        n = random.randint(0, nframes-1)
        pairs_train.append((os.path.join(folder_name, name_dict[folder_name][m]), os.path.join(folder_name, name_dict[folder_name][n])))

pairs_test = []
for folder_name in annos.keys():
    if folder_name not in ['B6Counting', 'B6Random']:
        continue    
    nframes = len([image for image in list(annos[folder_name].keys()) if 'SK_color' in image])
    print(nframes)
    for i in range(nframes):
        m = random.randint(0, nframes-1)
        n = random.randint(0, nframes-1)
        pairs_test.append((os.path.join(folder_name, name_dict[folder_name][m]), os.path.join(folder_name, name_dict[folder_name][n])))


# In[28]:


coords_dict_train = {}
coords_dict_test = {}
for folder_name in annos.keys():
    images = [image for image in list(annos[folder_name].keys()) if 'SK_color' in image]
    for img_name in images:
        anno = annos[folder_name][img_name]['uv_coord']
        if folder_name not in ['B6Counting', 'B6Random']:
            coords_dict_train[os.path.join(folder_name, img_name)] = anno
        else:
            coords_dict_test[os.path.join(folder_name, img_name)] = anno


# In[31]:


import pickle
with open('coords_dict_train.pkl', 'wb') as fp:
    pickle.dump(coords_dict_train, fp)
with open('pairs_train.pkl', 'wb') as fp:
    pickle.dump(pairs_train, fp)
with open('coords_dict_test.pkl', 'wb') as fp:
    pickle.dump(coords_dict_test, fp)
with open('pairs_test.pkl', 'wb') as fp:
    pickle.dump(pairs_test, fp)

