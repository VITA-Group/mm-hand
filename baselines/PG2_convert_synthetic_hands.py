
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import math
import h5py
import os
import sys

from skimage.morphology import square, dilation, erosion
import scipy

dic0 = json.load(open("/home/zhenyu/Desktop/datasets/Synthetic_Hands/synthetic_hands/fixed_view0/label_fixed_view0_2000.json","r"))
dic1 = json.load(open("/home/zhenyu/Desktop/datasets/Synthetic_Hands/synthetic_hands/fixed_view1/label_fixed_view1_2000.json","r"))
dic2 = json.load(open("/home/zhenyu/Desktop/datasets/Synthetic_Hands/synthetic_hands/fixed_view2/label_fixed_view2_2000.json","r"))
H_scale = 1920//1
W_scale = 1080//1
H_offset = 960//1
W_offset = 540//1
BUFF = 50
TARGET_SIZE = 256


# In[ ]:


def return_img_path(i):
    return "/home/zhenyu/Desktop/datasets/Synthetic_Hands/synthetic_hands/fixed_view2/%04d.png" % (i+1)


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

# def _sparse2dense(indices, values, shape):
#     dense = np.zeros(shape)
#     for i in range(len(indices)):
#         r = indices[i][0]
#         c = indices[i][1]
#         k = indices[i][2]
#         dense[r,c,k] = values[i]
#     return dense

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

def _oneDimSparsePose(indices, shape):
    ind_onedim = []
    for ind in indices:
        # idx = ind[2]*shape[0]*shape[1] + ind[1]*shape[0] + ind[0]
        idx = ind[0]*shape[2]*shape[1] + ind[1]*shape[2] + ind[2]
        ind_onedim.append(idx)
    shape = np.prod(shape)
    return ind_onedim, shape


# In[ ]:


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


# In[ ]:


def draw(keypoint_coord3d_v, index):
    height = 1920
    width = 1080

    
    path_img_out1 = "/home/zhenyu/Desktop/datasets/Synthetic_Hands/fixed_view/sparse_pose/" + str(index).zfill(4) + ".png"
    path_img_out2 = "/home/zhenyu/Desktop/datasets/Synthetic_Hands/fixed_view/pose_mask/" + str(index).zfill(4) + ".png"
    path_img_out3 = "/home/zhenyu/Desktop/datasets/Synthetic_Hands/fixed_view/image/" + str(index).zfill(4) + ".png"
    #path_img_out3 = "/home/zhenyu/Desktop/synthetic_hands_cropped/fixed_view2/" + str(index).zfill(4) + ".png"
    
    coords = np.array(keypoint_coord3d_v)[:,:2]
    coords[:, 0] = (coords[:, 0]-0.5)* W_scale+ W_offset
    coords[:, 1] = -(coords[:, 1]-0.5)* H_scale+ H_offset
    bounding_box = get_bbox(coords,W_scale,H_scale)

    img_crop, _ = crop_scale(cv2.imread(return_img_path(index)), bounding_box)
    img_resized = cv2.resize(img_crop, (TARGET_SIZE, TARGET_SIZE))
    cv2.imwrite(path_img_out3, img_resized)
    
    
    h_ratio, w_ratio = TARGET_SIZE / (bounding_box[3]-bounding_box[1]), TARGET_SIZE / (bounding_box[2]-bounding_box[0])
    coords[:, 0], coords[:, 1] = (coords[:, 0]-bounding_box[0])*w_ratio, (coords[:, 1]-bounding_box[1])*h_ratio
    
    indices_r4, values_r4, shape = _getSparsePose(coords, TARGET_SIZE, TARGET_SIZE, 22, radius=4, mode='Solid')
    pose_dense_r4 = _sparse2dense(indices_r4, values_r4, shape)
    print(pose_dense_r4.shape)
    cv2.imwrite(path_img_out1, _visualizePose(pose_dense_r4))

    pose_mask = _getPoseMask(coords, TARGET_SIZE, TARGET_SIZE, radius=8, var=4, mode='Solid')
    print(pose_mask.shape)
    cv2.imwrite(path_img_out2, _visualizePose(pose_mask))


# In[ ]:


import random
indices0 = list(range(len(dic0)))
random.shuffle(indices0)
print(indices0)
indices1 = list(range(len(dic1)))
random.shuffle(indices1)
print(indices1)
indices2 = list(range(len(dic2)))
random.shuffle(indices2)
print(indices2)


# In[ ]:


pairs = []
num = min(len(dic0), len(dic1), len(dic2))
for i in range(num):
    pairs.append((os.path.join(folder_path0, str(i).zfill(4)+'.png'), os.path.join(folder_path1, str(i).zfill(4)+'.png')))
    pairs.append((os.path.join(folder_path1, str(i).zfill(4)+'.png'), os.path.join(folder_path2, str(i).zfill(4)+'.png')))
    pairs.append((os.path.join(folder_path2, str(i).zfill(4)+'.png'), os.path.join(folder_path0, str(i).zfill(4)+'.png')))


# In[ ]:


coords_dict = {}
folder_path0 = '/home/zhenyu/Desktop/datasets/Synthetic_Hands/synthetic_hands/fixed_view0'
folder_path1 = '/home/zhenyu/Desktop/datasets/Synthetic_Hands/synthetic_hands/fixed_view1'
folder_path2 = '/home/zhenyu/Desktop/datasets/Synthetic_Hands/synthetic_hands/fixed_view2'
def coords_transform(coords_dict, dic, folder_path):
    for i in range(len(dic)):
        keypoint_coord3d_v = dic0[str(i).zfill(7)]['perspective']
        coords = np.array(keypoint_coord3d_v)[:,:2]
        coords[:, 0] = (coords[:, 0]-0.5)* W_scale+ W_offset
        coords[:, 1] = -(coords[:, 1]-0.5)* H_scale+ H_offset
        bounding_box = get_bbox(coords,W_scale,H_scale)
        h_ratio, w_ratio = TARGET_SIZE / (bounding_box[3]-bounding_box[1]), TARGET_SIZE / (bounding_box[2]-bounding_box[0])
        coords[:, 0], coords[:, 1] = (coords[:, 0]-bounding_box[0])*w_ratio, (coords[:, 1]-bounding_box[1])*h_ratio
        coords_dict[os.path.join(folder_path, str(i).zfill(4)+'.png')] = coords

coords_transform(coords_dict, dic0, folder_path0)
coords_transform(coords_dict, dic1, folder_path1)
coords_transform(coords_dict, dic2, folder_path2)


# In[ ]:


print(len(coords_dict))
print(coords_dict.keys())
labels = [1]*len(pairs)
_convert_dataset_one_pair_rec_withFlip(pairs, labels, coords_dict)

