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
from skimage.filters import gaussian
#from scipy.signal import savgol_filter
from math import  pi
from skimage.measure import moments, find_contours
from scipy.ndimage.measurements import center_of_mass

img_height, img_width = 960, 720
cls = 30
size = 340


target_dir = 'data//leaf//'

data_r = 'leaf_data_4_24_26_r.npy'
label_r = 'leaf_label_4_24_26_r.npy'

data = 'leaf_data_960_720.npy'
#data = 'leaf_data_480_360.npy'
label = 'leaf_label_480_360.npy'
leaf_data = np.load(target_dir+data)
#leaf_label = np.load(target_dir+'leaf_label_{}_{}.npy'.format(img_height, img_width))

m = 200
bins = 40
window = np.arange(-pi, pi, pi/m) # tune this carefully since dataset is small
leaf_data_CCD = -1* np.ones([len(leaf_data), len(window)])
leaf_data_curvature = -1* np.ones([len(leaf_data), len(window)])
leaf_CCD_hist = -1* np.ones([len(leaf_data), bins])

from scipy.interpolate import UnivariateSpline
import numpy as np

#------------------------------------------------------------------------------
# Calculate curvature
#------------------------------------------------------------------------------
def curvature_splines(r):

    t = window
#    std = error * np.ones(len(t))
    fr = UnivariateSpline(t, r, k = 3)

    rp = fr.derivative(1)(t)
    r2p = fr.derivative(2)(t)
    curvature = abs(r**2- r*r2p + 2*rp**2) / np.power(rp** 2 + r** 2, 3 / 2)
    return curvature


for ind, leaf in enumerate(leaf_data):
    dev = 0.5*len(np.where(leaf == False)[0])**0.25
    leaf = gaussian(leaf, sigma = dev)>0.7
    leaf_contour = ft.canny(leaf).astype(int)
    leaf_contour = max(find_contours(leaf, level = 0), key=len)
#    leaf_contour = np.array(leaf_contour)[0]
#    coord_x = np.where(leaf_contour == 1)[0]
#    coord_y = np.where(leaf_contour == 1)[1]
#    M = moments(leaf, order = 2)
#    c_x = M[1, 0] / M[0, 0]
#    c_y = M[0, 1] / M[0, 0]
#    c_x , c_y = np.mean(coord_x), np.mean(coord_y)
#    c_y, c_x = center_of_mass(leaf_contour)
    c_x, c_y = np.mean(leaf_contour, axis = 0)
    coord_x, coord_y = leaf_contour[:, 0], leaf_contour[:, 1]
    polar = np.zeros([len(coord_x), 2])
    polar[:,0] = ((coord_x -c_x)**2 + (coord_y - c_y)**2)**0.5
    polar[:,1] = np.arctan2( (coord_y - c_y), (coord_x - c_x) )
    polar = polar[polar[:,1].argsort()]
    r = interp1d(polar[:,1], polar[:,0], bounds_error = False, fill_value = polar[0,0])
    leaf_data_CCD[ind] = r(window)
    leaf_data_curvature[ind] = curvature_splines(r(window))
    leaf_CCD_hist[ind]= np.histogram(r(window), bins = bins, range = (20, 400), density = True)[0]    

def query_CCD(k):
    ind = np.where(leaf_label == k)
    sample = leaf_data_CCD[ind]
    sample_size = sample.shape[0]
    plt.figure()
    for i in range(5):
        for j in range(5):
            plt.subplot(5,5, 5*i+j+1)
            if 5*i+j < sample_size:
                plt.plot(sample[5*i+j]) 

if __name__ == '__main__':
    
    leaf_label = np.load(target_dir+label)
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
    curvature = np.zeros([cls, len(window)])
    
    for i in range(30):
        ind = np.where(leaf_label==i+1)[0]
        curve[i] = leaf_data_CCD[ind[0]]
        curve_fft[i] = leaf_fft[ind[0]]
        curvature[i] = leaf_data_curvature[ind[0]]
    
    plt.figure()
    for i in range(6):
        for j in range(5):
            plt.subplot(6,5, 5*i+j+1)
            plt.plot(window, curve[5*i+j])
    
    plt.figure()
    for i in range(6):
        for j in range(5):
            plt.subplot(6,5, 5*i+j+1)
            plt.plot(curve_fft[5*i+j, :])
    
    plt.figure()
    for i in range(6):
        for j in range(5):
            plt.subplot(6,5, 5*i+j+1)
            plt.plot(window, curvature[5*i+j])
    
    plt.figure()
    for i in range(6):
        for j in range(5):
            plt.subplot(6,5, 5*i+j+1)
            plt.hist(curve[5*i+j], bins=bins, range=(20, 400), normed = True)
    
    plt.figure()
    for i in range(6):
        for j in range(5):
            plt.subplot(6,5, 5*i+j+1)
            plt.hist(curvature[5*i+j])
            


