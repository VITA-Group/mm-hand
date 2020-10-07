
# coding: utf-8

# In[ ]:


import sys
print(sys.version)

import cv2
import glob, os
import numpy as np
import re
import fnmatch
import pickle
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os


# In[ ]:


import cv2
import glob, os
import numpy as np
import re
import fnmatch
import pickle
import random



def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def recursive_glob(rootdir='.', pattern='*'):
    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))

    return matches

def readAnnotation3D(file):
    f = open(file, "r")
    an = []
    for l in f:
        l = l.split()
        an.append((float(l[1]),float(l[2]), float(l[3])))

    return np.array(an, dtype=float)

def getCameraMatrix():
    Fx = 614.878
    Fy = 615.479
    Cx = 313.219
    Cy = 231.288
    cameraMatrix = np.array([[Fx, 0, Cx],
                    [0, Fy, Cy],
                    [0, 0, 1]])
    return cameraMatrix

def getDistCoeffs():
    return np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])

pathToDataset="../annotated_frames/"

cameraMatrix = getCameraMatrix()
distCoeffs = getDistCoeffs()

coord_dict_train = {}
coord_dict_test = {}
# iterate sequences
nframe_dict = {}
name_dict = {}
for i in [1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,19]:
    #if i in [12, 13]:
    #    continue
    # read the color frames
    if not os.path.exists("../cropped/data_"+str(i)+"/"):
        os.makedirs("../cropped/data_"+str(i)+"/")

    path = pathToDataset+"data_"+str(i)+"/"
    colorFrames = recursive_glob(path, "*_webcam_[0-9]*")
    colorFrames = natural_sort(colorFrames)
    print "There are",len(colorFrames),"color frames on the sequence data_"+str(i)
    # read the calibrations for each camera
    print "Loading calibration for ../calibrations/data_"+str(i)
    c_0_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_1/rvec.pkl","r"))
    c_0_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_1/tvec.pkl","r"))
    c_1_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_2/rvec.pkl","r"))
    c_1_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_2/tvec.pkl","r"))
    c_2_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_3/rvec.pkl","r"))
    c_2_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_3/tvec.pkl","r"))
    c_3_0 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_4/rvec.pkl","r"))
    c_3_1 = pickle.load(open("../calibrations/data_"+str(i)+"/webcam_4/tvec.pkl","r"))

    # rand_idx = random.randint(0, len(colorFrames))
    
    nframe_dict["data_"+str(i)] = len(colorFrames)
    name_dict["data_"+str(i)] = []
    for j in range(len(colorFrames)):
        #print colorFrames[j]
        toks1 = colorFrames[j].split("/")
        toks2 = toks1[3].split("_")
        jointPath = toks1[0]+"/"+toks1[1]+"/"+toks1[2]+"/"+toks2[0]+"_joints.txt"
        #print jointPath
        points3d = readAnnotation3D(jointPath)[0:21] # the last point is the normal


        webcam_id = int(toks2[2].split(".")[0])-1
        #print "Calibration for webcam id:",webcam_id
        if webcam_id == 0:
            rvec = c_0_0
            tvec = c_0_1
        elif webcam_id == 1:
            rvec = c_1_0
            tvec = c_1_1
        elif webcam_id == 2:
            rvec = c_2_0
            tvec = c_2_1
        elif webcam_id == 3:
            rvec = c_3_0
            tvec = c_3_1

        points2d, _ = cv2.projectPoints(points3d, rvec, tvec, cameraMatrix, distCoeffs)

        R,_ = cv2.Rodrigues(rvec)
        T = np.zeros((4,4))
        for l in range(R.shape[0]):
            for k in range(R.shape[1]):
                T[l][k] = R[l][k]

        for l in range(tvec.shape[0]):
            T[l][3] = tvec[l]
        T[3][3] = 1


        points3d_cam = []
        for k in range(len(points3d)):
            p = np.array(points3d[k]).reshape(3,1)
            p = np.append(p, 1).reshape(1,4)
            p_ = np.matmul(T, p.transpose())
            points3d_cam.append(p_)


        # compute the minimun Bounding box
        max_x = 0
        min_x = 99999
        max_y = 0
        min_y = 99999
        for k in range(len(points2d)):
            p = points2d[k][0]
            if p[0] > max_x:
                max_x = p[0]
            if p[0] < min_x:
                min_x = p[0]
            if p[1] > max_y:
                max_y = p[1]
            if p[1] < min_y:
                min_y = p[1]

        # compute the depth of the centroid of the joints
        p3d_mean = [0,0,0]
        for k in range(len(points3d_cam)):
            p3d_mean[0] += points3d_cam[k][0]
            p3d_mean[1] += points3d_cam[k][1]
            p3d_mean[2] += points3d_cam[k][2]
        p3d_mean[0] /= len(points3d_cam)
        p3d_mean[1] /= len(points3d_cam)
        p3d_mean[2] /= len(points3d_cam)


        # compute the offset considering the depth
        offset = 20 # 20px @ 390mm
        offset = p3d_mean[2]*offset/390
        max_x = int(max_x+offset)
        min_x = int(min_x-offset)
        max_y = int(max_y+offset)
        min_y = int(min_y-offset)

        if min_x < 0 or min_y < 0:
            continue

        img_pathToSave = "../cropped/data_"+str(i)+"/"+toks2[0]+"_img_"+toks2[2].split(".")[0]+".png"
        #print "Saving cropped and resized image", img_pathToSave
#         img = cv2.imread(colorFrames[j])
#             print(img.shape)
#         cropped_img = img[min_y:max_y, min_x:max_x]
#             print('ymax:{};ymin:{};ymax-ymin:{}'.format(max_y, min_y, max_y-min_y))
#             print('xmax:{};xmin:{};xmax-xmin:{}'.format(max_x, min_x, max_x-min_x))
#             print(cropped_img.shape)
#             print(colorFrames[j])
#         resized_img = cv2.resize(cropped_img, (256, 256))

#         cv2.imwrite(img_pathToSave, resized_img)

        orders = [20, 17, 16, 18, 19, 1, 0, 2, 3, 5, 4 ,6 ,7, 13 , 12, 14, 15, 9, 8, 10, 11]
#            anno_pathToSave = "../cropped/data_"+str(i)+"/"+toks2[0]+"_joints_"+toks2[2].split(".")[0]+".txt"
#             print "Saving hand joints", anno_pathToSave
#             with open(anno_pathToSave, "w") as f:
#                 for k in orders:
#                     f.write('{},{}\n'.format((int(points2d[k][0][1])-min_y)*256/(max_y-min_y),
#                                              (int(points2d[k][0][0])-min_x)*256/(max_x-min_x)))

        annos = []
        for k in orders:

            annos.append([(int(points2d[k][0][0])-min_x)*256/(max_x-min_x),
                                         (int(points2d[k][0][1])-min_y)*256/(max_y-min_y)])
        if i in [12,13]:
            coord_dict_test['/'.join([img_pathToSave.split('/')[-2], img_pathToSave.split('/')[-1]])] = np.asarray(annos)
        else:
            coord_dict_train['/'.join([img_pathToSave.split('/')[-2], img_pathToSave.split('/')[-1]])] = np.asarray(annos)
        
        name_dict["data_"+str(i)].append('/'.join([img_pathToSave.split('/')[-2], img_pathToSave.split('/')[-1]]))
#             img = cv2.imread("../cropped/data_"+str(i)+"/"+toks2[0]+"_img_"+toks2[2].split(".")[0]+".png")
#             os.remove("../cropped/data_"+str(i)+"/"+toks2[0]+"_img_"+toks2[2].split(".")[0]+".png")
#             print(img.shape)
#             for ind in range(len(orders)):
#                 k = orders[ind]
#                 print(k)
#                 cv2.circle(img, ((int(points2d[k][0][0])-min_x)*256/(max_x-min_x), 
#                                  (int(points2d[k][0][1])-min_y)*256/(max_y-min_y)), 3, (0,0,255))
#                 cv2.putText(img, str(ind), 
#                                 ((int(points2d[k][0][0])-min_x)*256/(max_x-min_x), 
#                                  (int(points2d[k][0][1])-min_y)*256/(max_y-min_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

#             cv2.imwrite("../cropped/data_"+str(i)+"/"+toks2[0]+"_annotated_img_"+toks2[2].split(".")[0]+".png", img)
#             print "Saving annotated hand joints to: " + "../bounding_boxes/data_"+str(i)+"/"+toks2[0]+"_annotated_img_"+toks2[2].split(".")[0]+".png"


# In[ ]:


print(nframe_dict)
print(name_dict['data_10'])


# In[ ]:


import random
print(name_dict)


# In[ ]:


train_pairs = []
for k, v in name_dict.items():
    if k in ['data_12', 'data_13']:
        continue
    nframes = len(name_dict[k])
    for i in range(nframes):
        m = random.randint(0, nframes-1)
        n = random.randint(0, nframes-1)
        train_pairs.append((name_dict[k][m], name_dict[k][n]))

test_pairs = []
for k, v in name_dict.items():
    if k not in ['data_12', 'data_13']:
        continue
    nframes = len(name_dict[k])
    for i in range(nframes):
        m = random.randint(0, nframes-1)
        n = random.randint(0, nframes-1)
        test_pairs.append((name_dict[k][m], name_dict[k][n]))

with open('realhands-pairs-train.csv',mode='w') as csv_file:
    fieldnames = ['from','to']
    writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames, escapechar=',', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for pair in train_pairs:
        writer.writerow({'from':pair[0], 'to':pair[1]})
        
with open('realhands-pairs-test.csv',mode='w') as csv_file:
    fieldnames = ['from','to']
    writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames, escapechar=',', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for pair in test_pairs:
        writer.writerow({'from':pair[0], 'to':pair[1]})


# In[ ]:


print(coord_dict_train.keys())


# In[ ]:


import csv

with open('mhp-annotation-train.csv',mode='w') as csv_file:
    fieldnames = ['name','keypoints_y','keypoints_x']
    writer = csv.DictWriter(csv_file, delimiter=':', fieldnames=fieldnames, escapechar=':', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for key, value in coord_dict_train.items():
        print(key)
        print(list(np.int_(value[:,1])))
        print(list(np.int_(value[:,0])))
        writer.writerow({'name':key, 
                         'keypoints_y':list(np.int_(value[:,1])),
                         'keypoints_x':list(np.int_(value[:,0]))})


# In[ ]:


with open('mhp-annotation-test.csv',mode='w') as csv_file:
    fieldnames = ['name','keypoints_y','keypoints_x']
    writer = csv.DictWriter(csv_file, delimiter=':', fieldnames=fieldnames, escapechar=':', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for key, value in coord_dict_test.items():
        print(key)
        print(list(np.int_(value[:,1])))
        print(list(np.int_(value[:,0])))
        writer.writerow({'name':key, 
                         'keypoints_y':list(np.int_(value[:,1])),
                         'keypoints_x':list(np.int_(value[:,0]))})

