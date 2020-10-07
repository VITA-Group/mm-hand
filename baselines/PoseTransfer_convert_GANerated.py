#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import random
import os
import csv
import numpy as np


# In[14]:


annos_train = pickle.load(open("GANerated/train/annotation.pickle", "rb"))
print(annos_train.keys())
annos_test = pickle.load(open("GANerated/test/annotation.pickle", "rb"))
print(annos_test.keys())


# In[18]:


test_name_dict = {}
for folder_name in annos_test.keys():
    test_name_dict[folder_name] = []
    for img_name in annos_test[folder_name]:
        test_name_dict[folder_name].append(img_name)

test_pairs = []
for folder_name in annos_test.keys(): 
    nframes = len(list(annos_test[folder_name].keys()))
    print(nframes)
    for i in range(nframes):
        m = random.randint(0, nframes-1)
        n = random.randint(0, nframes-1)
        test_pairs.append((os.path.join(folder_name, test_name_dict[folder_name][m]), os.path.join(folder_name, test_name_dict[folder_name][n])))

train_name_dict = {}
for folder_name in annos_train.keys():
    train_name_dict[folder_name] = []
    for img_name in annos_train[folder_name]:
        train_name_dict[folder_name].append(img_name)

train_pairs = []
for folder_name in annos_train.keys(): 
    nframes = len(list(annos_train[folder_name].keys()))
    print(nframes)
    for i in range(nframes):
        m = random.randint(0, nframes-1)
        n = random.randint(0, nframes-1)
        train_pairs.append((os.path.join(folder_name, train_name_dict[folder_name][m]), os.path.join(folder_name, train_name_dict[folder_name][n])))


# In[ ]:


with open('ganerated-pairs-train.csv',mode='w') as csv_file:
    fieldnames = ['from','to']
    writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames, escapechar=',', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for pair in train_pairs:
        writer.writerow({'from':pair[0], 'to':pair[1]})
        
with open('ganerated-pairs-test.csv',mode='w') as csv_file:
    fieldnames = ['from','to']
    writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames, escapechar=',', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for pair in test_pairs:
        writer.writerow({'from':pair[0], 'to':pair[1]})


# In[21]:


coord_dict_train = {}
coord_dict_test = {}
for folder_name in annos_train.keys():
    images = list(annos_train[folder_name].keys())
    for img_name in images:
        anno = annos_train[folder_name][img_name]['uv_coord']
        coord_dict_train[os.path.join(folder_name, img_name)] = anno
        
for folder_name in annos_test.keys():
    images = list(annos_test[folder_name].keys())
    for img_name in images:
        anno = annos_test[folder_name][img_name]['uv_coord']
        coord_dict_test[os.path.join(folder_name, img_name)] = anno


# In[22]:


with open('ganerated-annotation-train.csv',mode='w') as csv_file:
    fieldnames = ['name','keypoints_y','keypoints_x']
    writer = csv.DictWriter(csv_file, delimiter=':', fieldnames=fieldnames, escapechar=':', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for key, value in coord_dict_train.items():
        writer.writerow({'name':key, 
                         'keypoints_y':list(np.int_(value[:,1])),
                         'keypoints_x':list(np.int_(value[:,0]))})

with open('ganerated-annotation-test.csv',mode='w') as csv_file:
    fieldnames = ['name','keypoints_y','keypoints_x']
    writer = csv.DictWriter(csv_file, delimiter=':', fieldnames=fieldnames, escapechar=':', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for key, value in coord_dict_test.items():
        writer.writerow({'name':key, 
                         'keypoints_y':list(np.int_(value[:,1])),
                         'keypoints_x':list(np.int_(value[:,0]))})

