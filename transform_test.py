# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:08:59 2017

@author: dykua
"""

from skimage import io
import numpy as np
from scipy import optimize
from skimage.transform import resize, warp, rotate, rescale, AffineTransform
import matplotlib.pyplot as plt
import os

def addpath(datapath = 'data//leaf//BW//23. Erodium sp//'):
    
    print('Data path added.\n')
    return datapath

def dist(imgA, imgB):
    return np.sum((imgA - imgB)**2)/(imgA.shape[0]*imgA.shape[1])

def loss(imgA, imgB, A_inv):
    return dist(imgA, imgB)

def loss_corr(imgA, imgB, A_inv):
    return 1-np.corrcoef(imgA.flatten(), imgB.flatten())[0,1]

def reduce_size(img, ratio = 10):
    img_height, img_width = img.shape
    img_height, img_width = int(img_height/ratio), int(img_width/ratio)
    img_reduced = resize(img, (img_height, img_width))
    return img_reduced, img_height, img_width

'''
Q1: different optimization for a whole matrix 
or
each optimization for specific transform?

Q2: The rotation center

tform1 + tform2 is composible

Can use certain landmarks to estimate transformation
'''
def Transform_mat(tmatrix): # tmatrix is a 3 by 2 matrix
    proj_matrix = np.column_stack((tmatrix,np.array([0,0,1]))).transpose() # a transpose?
#    print(proj_matrix)
    tform = AffineTransform(matrix = proj_matrix)
    return tform

def Transform_num(translation = (0,0),
                  scale = (1,1),
                  rotation = 0):
    return AffineTransform(scale = scale, rotation = rotation, translation = translation)

'''
TODO:
    1. Add reflection into the transformation. This can be done by flipping a numpy array
    2. The scale is wrt (0,0), not the center of the image. rotation is..
'''

def Transform_seq(img,
                  translation = (0,0),
                  scale = (1,1),
                  rotation = 0):   
    img_warped = warp(img, AffineTransform(translation = translation),mode = 'constant', cval = 1.0)
    img_warped = warp(img_warped, AffineTransform(scale = scale), mode = 'constant', cval = 1.0)
#    img_warped = rescale(img_warped, scale = scale, mode = 'constant', cval = 1.0)
    img_warped = rotate(img_warped, angle = rotation, mode = 'constant', cval = 1.0) # angle uses degree
    
    return img_warped

def registration(img_moving, img_fixed, disp = True, same_scale = False): #warp the moving imag to the ref img
    
    if same_scale is False:            
        res = optimize.minimize(lambda x: loss(Transform_seq(img_moving, translation = (x[0],x[1]),
                                                         scale = (x[2],x[3]),
                                                         rotation = x[4]), 
                                            img_fixed, x),
                                            np.array([0,0,1,1,0]),
                                            method='Powell',
                                            tol = 1e-6,
                                            options={'disp': disp})   
    
        return Transform_seq(img_moving, 
                         translation = (res.x[0],res.x[1]), 
                         scale = (res.x[2],res.x[3]),
                         rotation = res.x[4]), \
                             res.x, res.fun
    else:
        res = optimize.minimize(lambda x: loss(Transform_seq(img_moving, translation = (x[0],x[1]),
                                                         scale = (x[2],x[2]),
                                                         rotation = x[3]), 
                                            img_fixed, x),
                                            np.array([0,0,1,1]),
                                            method='Powell',
                                            tol = 1e-6,
                                            options={'disp': disp})   
    
        return Transform_seq(img_moving, 
                         translation = (res.x[0],res.x[1]), 
                         scale = (res.x[2],res.x[2]),
                         rotation = res.x[3]), \
                             res.x, res.fun
           


'''
TODO: What to minimize? How to corporate vector fields?
''' 
#==============================================================================
# This function calculates the total distance from current image average to 
# each image sample in img
#==============================================================================
def obj_func_for_ave(Phi, img_ini, img):
    samples= img.shape[0]
    img_height, img_width = img_ini.shape
    img_warped_I = np.ones((samples, img_height, img_width))
    warp_I = np.zeros((samples,5))
    dist_I = np.ones(samples)
#    Vec_Field_I = np.ones((2, img_height, img_width, samples))
    img_ave = Transform_seq(img_ini, 
                            translation = (Phi[0],Phi[1]), 
                            scale = (Phi[2],Phi[3]),
                            rotation = Phi[4])
    for i in range(samples):
        img_warped_I[i], warp_I[i], dist_I[i]  = registration(img[i], img_ave, disp = False)

                    
    return np.sum(dist_I)
  
    
if __name__=='__main__':
    datapath = addpath()
    ref_img = io.imread(datapath+'iPAD2_C23_EX11_B.tiff').astype(float)
    ref_img, _, _ = reduce_size(ref_img)
#    trans_matrix = np.array([[1,0],[0,1],[10,10]]) # The transformation is with respect to the origin, NOT center
#    trans = Transform(trans_matrix)
#    img_warped = warp(ref_img, trans, mode = 'constant', cval = 1.0)
#    io.imshow(img_warped)
    
#    io.imshow(Transform_seq(ref_img, rotation = 2))

#    img_warped = Transform_seq(ref_img, rotation = 20, scale = (1.1, 1.1), translation = (5,10))
##    print(loss_corr(ref_img, img_warped, [1,0]))
#    res = optimize.minimize(lambda x: loss_corr(Transform_seq(img_warped, translation = (x[0],x[1]),
#                                                         scale = (x[2],x[2]),
#                                                         rotation = x[3]), 
#                                            ref_img, x),
#                                np.array([0,0,1,1]),
#                                method='Powell',
#                                tol = 1e-4,
#                                options={'disp': True})
#    
#    figg = plt.figure()
#    figg.add_subplot(131)
#    io.imshow(ref_img)
#    figg.add_subplot(132)
#    io.imshow(img_warped)
#    figg.add_subplot(133)
#    io.imshow(Transform_seq(img_warped, 
#                            translation = (res.x[0],res.x[1]), 
#                            scale = (res.x[2],res.x[2]),
#                            rotation = res.x[3]))
    

#==============================================================================
# Test warp between two actual template
#==============================================================================
    
    tar_img = io.imread(datapath+'iPAD2_C23_EX01_B.tiff').astype(float)
    tar_img, _, _ = reduce_size(tar_img)
    img_warped, res,_ = registration(tar_img, ref_img)
    
    figg = plt.figure()
    figg.add_subplot(131)
    io.imshow(ref_img)
    figg.add_subplot(132)
    io.imshow(tar_img)
    figg.add_subplot(133)
    io.imshow(img_warped)
#    


#==============================================================================
# Test for the group average
#==============================================================================    
    
#    img_ini = io.imread(datapath + 'iPAD2_C35_EX10_B.tiff').astype(float)
#    img_ini, img_height, img_width = reduce_size(img_ini)
#    img = np.ones((11, img_height, img_width))
#    img_ind = 0
#    for file in sorted(os.listdir(datapath)):
#        img[img_ind],_,_ = reduce_size(io.imread(datapath+file).astype(float))
#        img_ind += 1
#        
#    res = optimize.minimize(lambda x: obj_func_for_ave(x, img_ini, img[:,:,:]), 
#                            np.array([0,0,1,1,0]), 
#                            method = 'Powell',
#                            tol = 1e-2,
#                            options={'disp': True})
#    
#    fig=plt.figure()
#    io.imshow(Transform_seq(img_ini, 
#                            translation = (res.x[0],res.x[1]), 
#                            scale = (res.x[2],res.x[3]),
#                            rotation = res.x[4]))
#    plt.title("The average leaf")