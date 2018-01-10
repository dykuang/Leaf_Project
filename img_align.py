# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:54:10 2017

@author: dykua

This scripts reads pre-stored .npy leaf image file and align each group to the
first member of the group
"""

from transform_test import registration
import numpy as np

img_height, img_width = 64, 48
cls = 30
size = 340

target_dir = 'data//leaf//'
leaf_data = np.load(target_dir+'leaf_data_{}_{}.npy'.format(img_height, img_width))
leaf_label = np.load(target_dir+'leaf_label_{}_{}.npy'.format(img_height, img_width))

leaf_data_aligned = np.ones(leaf_data.shape)
#dist = -1*np.ones(size)
count = 0
for i in range(cls):
    cls_ind = np.where(leaf_label==i+1)[0]
    leaf_x = leaf_data[cls_ind]
    for leaf in leaf_x:
        img, a, b = registration(leaf, leaf_x[0], disp = False, same_scale = False)
        leaf_data_aligned[count,:,:] = img
        count = count + 1
        
np.save(target_dir +'leaf_data_{}_{}_aligned_diff_scale.npy'.format(img_height, img_width),
        leaf_data_aligned)