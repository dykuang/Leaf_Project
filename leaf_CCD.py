# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:56:57 2017

@author: dykua

This script extracts CCD feature from leaf data
"""

#import matplotlib.pyplot as plt
import numpy as np
from skimage import feature as ft
#from skimage.measure import find_contours
from scipy.interpolate import interp1d
#from scipy.signal import savgol_filter
from math import  pi



img_height, img_width = 480, 360
cls = 30
size = 340


target_dir = 'data//leaf//'
leaf_data = np.load(target_dir+'leaf_data_{}_{}.npy'.format(img_height, img_width))
#leaf_label = np.load(target_dir+'leaf_label_{}_{}.npy'.format(img_height, img_width))

window = np.arange(-pi+pi/25, pi-pi/25, pi/25) # tune this carefully since dataset is small
leaf_data_CCD = -1* np.ones([len(leaf_data), len(window)])
for ind, leaf in enumerate(leaf_data):
    leaf_contour = ft.canny(leaf, sigma = 2)
    coord_x = np.where(leaf_contour == 1)[0]
    coord_y = np.where(leaf_contour == 1)[1]
    c_x , c_y = np.mean(coord_x), np.mean(coord_y)
    polar = np.zeros([len(coord_x), 2])
    polar[:,0] = ((coord_x -c_x)**2 + (coord_y - c_y)**2)**0.5
    polar[:,1] = np.arctan2( (coord_y - c_y), (coord_x - c_x) )
    polar = polar[polar[:,1].argsort()]
    r = interp1d(polar[:,1], polar[:,0])
    leaf_data_CCD[ind] = r(window)

if __name__ == '__main__':
    
    leaf_label = np.load(target_dir+'leaf_label_{}_{}.npy'.format(img_height, img_width))
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    from scipy.signal import blackman
    
    N = leaf_data_CCD.shape[1]
    w = blackman(N)
    
    leaf_fft = np.zeros([len(leaf_data_CCD), N//2])
    for i, leaf in enumerate(leaf_data_CCD):
       leaf = fft(leaf*w)
       leaf_fft[i] = 2*np.abs(leaf[:N//2])/N
               
    curve = np.zeros([cls, len(window)])
    curve_fft = np.zeros([cls, N//2])
    
    for i in range(30):
        ind = np.where(leaf_label==i+1)[0]
        curve[i] = leaf_data_CCD[ind[0]]
        curve_fft[i] = leaf_fft[ind[0]]
    
    plt.figure()
    for i in range(6):
        for j in range(5):
            plt.subplot(6,5, 5*i+j+1)
            plt.plot(window, curve[5*i+j])
    
    plt.figure()
    for i in range(6):
        for j in range(5):
            plt.subplot(6,5, 5*i+j+1)
            plt.plot(curve_fft[5*i+j, :6])
            


