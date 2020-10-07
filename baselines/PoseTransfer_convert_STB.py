#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pickle
annos = pickle.load(open("STB_cropped/train/stb_annotation_train.pickle", "rb"))
print(len(list(annos.keys())))
print(list(annos.keys()))


# In[5]:


import os
from shutil import copyfile
import cv2
import random
import csv
import numpy as np

if not os.path.exists('STB_cropped'):
    os.makedirs('STB_cropped')



name_dict = {}
for folder_name in annos.keys():
    print(folder_name)
    name_dict[folder_name] = []
    for img_name in annos[folder_name]:
        if 'SK_color' in img_name:
            name_dict[folder_name].append(img_name)
print(len(name_dict['B6Random']))


# In[9]:


train_pairs = []
for folder_name in annos.keys():
    if folder_name in ['B6Counting', 'B6Random']:
        continue    
    nframes = len([image for image in list(annos[folder_name].keys()) if 'SK_color' in image])
    print(nframes)
    for i in range(nframes):
        m = random.randint(0, nframes-1)
        n = random.randint(0, nframes-1)
        train_pairs.append((os.path.join(folder_name, name_dict[folder_name][m]), os.path.join(folder_name, name_dict[folder_name][n])))

test_pairs = []
for folder_name in annos.keys():
    if folder_name not in ['B6Counting', 'B6Random']:
        continue    
    nframes = len([image for image in list(annos[folder_name].keys()) if 'SK_color' in image])
    print(nframes)
    for i in range(nframes):
        m = random.randint(0, nframes-1)
        n = random.randint(0, nframes-1)
        test_pairs.append((os.path.join(folder_name, name_dict[folder_name][m]), os.path.join(folder_name, name_dict[folder_name][n])))
        
with open('stb-pairs-train.csv',mode='w') as csv_file:
    fieldnames = ['from','to']
    writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames, escapechar=',', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for pair in train_pairs:
        writer.writerow({'from':pair[0], 'to':pair[1]})
        
with open('stb-pairs-test.csv',mode='w') as csv_file:
    fieldnames = ['from','to']
    writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames, escapechar=',', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for pair in test_pairs:
        writer.writerow({'from':pair[0], 'to':pair[1]})


coord_dict_train = {}
coord_dict_test = {}
for folder_name in annos.keys():
    images = [image for image in list(annos[folder_name].keys()) if 'SK_color' in image]
    for img_name in images:
        anno = annos[folder_name][img_name]['uv_coord']
        if folder_name not in ['B6Counting', 'B6Random']:
            coord_dict_train[os.path.join(folder_name, img_name)] = anno
        else:
            coord_dict_test[os.path.join(folder_name, img_name)] = anno

with open('stb-annotation-train.csv',mode='w') as csv_file:
    fieldnames = ['name','keypoints_y','keypoints_x']
    writer = csv.DictWriter(csv_file, delimiter=':', fieldnames=fieldnames, escapechar=':', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for key, value in coord_dict_train.items():
        writer.writerow({'name':key, 
                         'keypoints_y':list(np.int_(value[:,1])),
                         'keypoints_x':list(np.int_(value[:,0]))})

with open('stb-annotation-test.csv',mode='w') as csv_file:
    fieldnames = ['name','keypoints_y','keypoints_x']
    writer = csv.DictWriter(csv_file, delimiter=':', fieldnames=fieldnames, escapechar=':', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for key, value in coord_dict_test.items():
        writer.writerow({'name':key, 
                         'keypoints_y':list(np.int_(value[:,1])),
                         'keypoints_x':list(np.int_(value[:,0]))})

