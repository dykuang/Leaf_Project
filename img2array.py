# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:27:02 2017

@author: dykua

This script read the data, reduce its size and then stack them as numpy array
for latter training purposes
"""

import os
from skimage import io
import numpy as np
from skimage.transform import resize

target_dir = 'data//leaf//BW//'
save_dir = 'data//leaf//'

img_height, img_width = int(256), int(256)
size = 340


              
leaf_data = np.zeros([size, img_height, img_width])
leaf_label = np.zeros(size)
prof_name = []

ind = 0
for cls ,[root, dirs, files]in enumerate(os.walk(target_dir)):
    for sample, file in enumerate(files):
        path = os.path.join(root, file)

        leaf_data[ind] = resize(io.imread(path,as_grey = True).astype(float),
                     (img_height, img_width), mode = 'reflect')
        
            
        leaf_label[ind] = int(cls)
        
        ind = ind + 1
    
    prof_name.append(root)
     
print('%d files converted.' % leaf_data.shape[0])  
filename = save_dir+'leaf_data_{}_{}'.format(img_height, img_width)
np.save(filename, leaf_data)
#labelname = save_dir+'leaf_label_{}_{}'.format(img_height, img_width)
#np.save(labelname, leaf_label)



