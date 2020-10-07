#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pickle
import random
import os
import csv
import numpy as np


# In[45]:


annos_train = pickle.load(open("RHD/rhd_cropped/training/annotation.pickle", "rb"))
train_pairs = []
n_images = len(list(annos_train['color'].keys()))
print(n_images)
images = list(annos_train['color'].keys())
for i in range(n_images):
    m = random.randint(0, n_images - 1)
    n = random.randint(0, n_images - 1)
    train_pairs.append((os.path.join('color', images[m]), os.path.join('color', images[n])))


# In[46]:


annos_test = pickle.load(open("RHD/rhd_cropped/evaluation/annotation.pickle", "rb"))
test_pairs = []
n_images = len(list(annos_test['color'].keys()))
print(n_images)
images = list(annos_test['color'].keys())
for i in range(n_images):
    m = random.randint(0, n_images - 1)
    n = random.randint(0, n_images - 1)
    test_pairs.append((os.path.join('color', images[m]), os.path.join('color', images[n])))


# In[47]:


with open('rhd-pairs-train.csv',mode='w') as csv_file:
    fieldnames = ['from','to']
    writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames, escapechar=',', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for pair in train_pairs:
        writer.writerow({'from':pair[0], 'to':pair[1]})
        
with open('rhd-pairs-test.csv',mode='w') as csv_file:
    fieldnames = ['from','to']
    writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=fieldnames, escapechar=',', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for pair in test_pairs:
        writer.writerow({'from':pair[0], 'to':pair[1]})


# In[48]:


coord_dict_train = {}
coord_dict_test = {}


# In[49]:


for img_name in annos_train['color'].keys():
    anno = annos_train['color'][img_name]['uv_coord']
    coord_dict_train[os.path.join('color', img_name)] = anno


# In[50]:


for img_name in annos_test['color'].keys():
    anno = annos_test['color'][img_name]['uv_coord']
    coord_dict_test[os.path.join('color', img_name)] = anno


# In[51]:


with open('rhd-annotation-train.csv',mode='w') as csv_file:
    fieldnames = ['name','keypoints_y','keypoints_x']
    writer = csv.DictWriter(csv_file, delimiter=':', fieldnames=fieldnames, escapechar=':', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for key, value in coord_dict_train.items():
        writer.writerow({'name':key, 
                         'keypoints_y':list(np.int_(value[:,1])),
                         'keypoints_x':list(np.int_(value[:,0]))})

with open('rhd-annotation-test.csv',mode='w') as csv_file:
    fieldnames = ['name','keypoints_y','keypoints_x']
    writer = csv.DictWriter(csv_file, delimiter=':', fieldnames=fieldnames, escapechar=':', quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for key, value in coord_dict_test.items():
        writer.writerow({'name':key, 
                         'keypoints_y':list(np.int_(value[:,1])),
                         'keypoints_x':list(np.int_(value[:,0]))})

