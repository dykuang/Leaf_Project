# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:59:40 2018

@author: Dongyang

This script tests classifiers on swidish leaf data set

By stacking CCD, FFT, HIST and AutoCorr, the accuracy can reach 94%.
FFT + AutoCorr can be 1% or 2% percentage less.

By data augmentation, hyperparameter tuning, stacking pretrained network,

the best performance now is 96.75% with the residual architure + pretrain
can not recover there after.....
"""

from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, BatchNormalization, add
from keras.models import Model
from keras import backend as K
from keras import optimizers, losses, utils
from sklearn.preprocessing import  RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder, Normalizer, QuantileTransformer
from sklearn.metrics import accuracy_score
import numpy as np
import random
from keras.callbacks import ModelCheckpoint, EarlyStopping

cls = 15 
size = 75 * 15
batchsize = 20
epochs = 20

target_dir = 'data//leaf//'

leaf_data = np.zeros((size, 200))
leaf_label = np.zeros(size)

#------------------------------------------------------------------------------
# Read data
#------------------------------------------------------------------------------
for i in range(cls):
    leaf_data[i*75:(i+1)*75] = np.load(target_dir + 'S_leaf_CCD{}.npy'.format(i+1))
#    leaf_data[i*75:(i+1)*75] = np.load(target_dir + 'S_leaf_CCD_nearest{}.npy'.format(i+1))
#    leaf_data[i*75:(i+1)*75] = np.load(target_dir + 'S_leaf_CCD_quad{}.npy'.format(i+1)) # worse because of the high spike.
    leaf_label[i*75:(i+1)*75] = i
    


# =============================================================================
# data = 'leaf_data_CCD_cv.npy'
# label = 'leaf_label_480_360.npy'
# leaf_data = np.load(target_dir + data)
# leaf_label = np.load(target_dir+ label)
# leaf_label = leaf_label -1
# cls = 30
# size = 340
# =============================================================================

# =============================================================================
# 
# import csv
# target = r'data/100 leaves plant species/data_Mar_64.txt'
# 
# leaf_Mar = []
# with open(target) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         leaf_Mar.append(row)
#              
# leaf_Mar = np.asarray(leaf_Mar)
# leaf_Mar = leaf_Mar[:,1:].astype(float)
# 
# target = r'data/100 leaves plant species/data_Sha_64.txt'
# leaf_Sha = []
# with open(target) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         leaf_Sha.append(row)         
# leaf_Sha = np.asarray(leaf_Sha)
# leaf_Sha = leaf_Sha[:,1:].astype(float)
# 
# 
# target = r'data/100 leaves plant species/data_Tex_64.txt'
# 
# leaf_Tex = []
# with open(target) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         leaf_Tex.append(row)
#              
# leaf_Tex = np.asarray(leaf_Tex)
# leaf_Tex = leaf_Tex[:,1:].astype(float)
# 
# leaf_label = np.zeros([1600])
# for i in range(100):
#     leaf_label[16*i:16*i+15] = i
#     
# cls=100
# size = 1599
# 
# leaf_Sha= np.vstack([leaf_Sha[:15], leaf_Sha[16:]])
# leaf_Mar= np.vstack([leaf_Mar[:15], leaf_Mar[16:]])
# leaf_label = np.hstack([leaf_label[:15], leaf_label[16:]])
# leaf_data = np.hstack([leaf_Sha, leaf_Mar , leaf_Tex])
# =============================================================================

#leaf_data = leaf_Sha 
#==============================================================================
# Moving average
#==============================================================================
def movingaverage (series, window):
    sma = np.zeros((series.shape[0], series.shape[1]-window+1))
    weights = np.repeat(1.0, window)/window
    for i, s in enumerate(series):
        sma[i] = np.convolve(s, weights, 'valid')
    return sma
 
#------------------------------------------------------------------------------
# Second level feature extraction
#------------------------------------------------------------------------------
from scipy.fftpack import fft
from scipy.signal import blackman

def curvefft(curve_data):
    N = curve_data.shape[1]
    w = blackman(N)  # does not seems to matter much
    
    curve_fft = np.zeros([len(curve_data), N//2])
    for i, curve in enumerate(curve_data):
       curve = fft(w*(curve - np.mean(curve)))
       curve_fft[i] = 2*np.abs(curve[:N//2])/N
       
    return (curve_fft)

def leaf_hist(data, bins=30, density=False):
    hist = np.zeros((data.shape[0], bins))
    for i, ccd in enumerate(data):
        hist[i] = np.histogram(ccd, bins=bins, density = density)[0]
        
    return hist

import pandas as pd
#from statsmodels.tsa.stattools import pacf, acf
def auto_corr(data, lag = 2):
    N = int(data.shape[1]/lag) - 1
    ac = np.zeros((data.shape[0], N))
    for i, ccd in enumerate(data):
        for j in range(N):
            ac[i,j] = pd.Series(ccd - np.mean(ccd)).autocorr((j+1)*lag)
    return ac

#def auto_corr_v2(data, maxlag = 50):
#    ac = np.zeros((data.shape[0], maxlag))
#    for i, ccd in enumerate(data):
#        ac[i] = acf(ccd - np.mean(ccd), nlags = maxlag)[1:]
#    return ac
#
#def par_acf(data, maxlag = 50):
#    pac = np.zeros((data.shape[0], maxlag))
#    for i, ccd in enumerate(data):
#        pac[i] = pacf(ccd - np.mean(ccd), nlags = maxlag)[1:]
#    return pac


from pywt import dwt


def pydwt(data):
    CA, CD = [], []
    for ccd in data:
        cA, cD = dwt(ccd, 'db1')
        CA.append(cA)
        CD.append(cD)
    return [np.vstack(CA), np.vstack(CD)]


def preprocess(train, test, flag = True):
    if True:
#        scaler = StandardScaler().fit(train)
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
    return train, test

from sklearn.decomposition import PCA
def addpca(train, test, comp = 40):
    pre_pca = PCA(n_components=comp).fit(train)  # using others than pca?
    x_train = pre_pca.transform(train)
    x_test = pre_pca.transform(test)
    
    return x_train, x_test

#==============================================================================
# Train/test split
#==============================================================================

#leaf_data = movingaverage(leaf_data, 3)
#leaf_data = leaf_data[:, ::3]
#       

'''
test size must be reasonably large to show "average" performance
'''

x_train, x_test, y_train, y_test, ind_train, ind_test = train_test_split(
                             leaf_data, leaf_label, np.arange(size), 
                             test_size=0.25, 
                             random_state = 233,
                             shuffle = True, stratify = leaf_label)

# augment the data
'''
The augmentation seems to help bring up 1% or 2%
You can try adding more "corruptions"
'''
def datagen(train, label, 
            flip = True, 
            shift = True,
            shift_percent = 0.4,
            shift_range = 0.2):
    
    if flip:
        train = np.vstack((train, np.flip(x_train, axis =1 )))
        label = np.hstack((label, label))
    
    if shift:
        N, M = train.shape
        num_shift = np.floor(N*shift_percent).astype(int)
        shift_unit = np.floor(M*shift_range).astype(int)

        ind = random.sample(range(1,N), num_shift)
        shifted = np.zeros((len(ind), M))
        label_shift = np.zeros(len(ind))
        for i, ii in enumerate(ind):
            shifted[i]= np.roll(train[ii], 
                                random.randint(-shift_unit, shift_unit))
            label_shift[i] = label[ii]
            
        train = np.vstack((train, shifted))
        label = np.hstack((label, label_shift))
    
    return train, label

aug_flag = True
if aug_flag:
    x_train = np.vstack((x_train, 
                         np.flip(x_train, axis = 1),
                         np.roll(x_train, 10, axis = 1),
                         np.roll(x_train, 5, axis = 1),
                         np.roll(x_train, -5, axis = 1)
                         ))
    
    y_train = np.hstack((y_train, y_train, y_train, 
                         y_train, y_train))
#
    
#    x_train, y_train = datagen(x_train, y_train)

#        
#x_train = (x_train - np.mean(x_train, axis=1).reshape(len(x_train),1))/np.max(x_train, axis = 1). reshape(len(x_train),1)   
#x_test = (x_test - np.mean(x_test, axis=1).reshape(len(x_test),1))/np.max(x_test, axis = 1). reshape(len(x_test),1)   



#x_train = (x_train - np.mean(x_train, axis = 0))/np.max(x_train)
#x_test = (x_test - np.mean(x_train, axis = 0))/np.max(x_train)

#x_train = np.hstack((x_train, 
#                     curvefft(x_train)[:,:], 
#                     leaf_hist(x_train)))
#
#x_test = np.hstack((x_test, 
#                    curvefft(x_test)[:,:], 
#                    leaf_hist(x_test)))

#x_train = np.load(target_dir + 'Swedish_train.npy')
#x_test = np.load(target_dir + 'Swedish_test.npy')
#x_train = x_train[:,1:]
#x_test = x_test[:,1:]


#x_train, x_test = preprocess(x_train, x_test)
#x_train_fft, x_test_fft = preprocess(curvefft(x_train), curvefft(x_test))
#x_train_acf, x_test_acf = preprocess(auto_corr(x_train), auto_corr(x_test))

#x_train_stack= np.hstack((x_train,
#                     x_train_fft, 
#                     x_train_acf
#
#                     ))
#
#x_test_stack= np.hstack((x_test,
#                    x_test_fft, 
#                    x_test_acf
#
#                    ))

'''
replacing x_train with cwt's ca can give comparable results
features to stack: raw, fft, acf, hist, cwt's ca, cwt's cd

Does not seem to be able to break the bottleneck by stacking more.
'''

x_train_stack= np.hstack((x_train,
                     curvefft(x_train)[:,:], 
                     auto_corr(x_train,2)

                     ))

x_test_stack= np.hstack((x_test,
                    curvefft(x_test)[:,:], 
                    auto_corr(x_test,2)

                    ))


#
#x_train_stack = x_train
#x_test_stack = x_test


#==============================================================================
# Select a template for each class
#==============================================================================
#template = -1*np.ones(30)
#for i in range(30):
#    if i+1 not in exclude:
#        template[i] = ind_train[np.where(y_train == i)][0]
    

#------------------------------------------------------------------------------
#Preprocess: normalization, pca, kernel pca?
#------------------------------------------------------------------------------
y_train = utils.to_categorical(y_train, cls)
y_test = utils.to_categorical(y_test, cls)

# normalization
scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train_stack)
#scaler = QuantileTransformer().fit(x_train_stack)
#scaler = StandardScaler().fit(x_train_stack)
x_train_std = scaler.transform(x_train_stack) 
x_test_std = scaler.transform(x_test_stack)

# PCA
#from sklearn.decomposition import PCA
'''
PCA components at around 40 gives comparable accuracy
'''
#pre_pca = PCA(n_components=40).fit(x_train_std) 
#x_train_std = pre_pca.transform(x_train_std)
#x_test_std = pre_pca.transform(x_test_std)

#x_train_std = np.hstack((x_train_std, auto_corr(x_train)))
#x_test_std = np.hstack((x_test_std, auto_corr(x_test)))

#------------------------------------------------------------------------------
# Using features from COTES
#------------------------------------------------------------------------------
#x_train_std = np.load(target_dir + 'Swedish_train.npy')
#x_test_std = np.load(target_dir + 'Swedish_test.npy')
#
#x_train_std = x_train_std[:,1:]
#x_test_std = x_test_std[:,1:]
#
#scaler = StandardScaler().fit(x_train_std)
#x_train_std = scaler.transform(x_train_std) 
#x_test_std = scaler.transform(x_test_std)
#
#y_train = utils.to_categorical(x_train_std[:,0], cls)
#y_test = utils.to_categorical(x_test_std[:,0], cls)
#------------------------------------------------------------------------------
from keras.layers.advanced_activations import PReLU
input_dim = x_train_std.shape[1]

feature = Input(shape= (input_dim, ) )
#feature1 = Dropout(0.25)(feature)
x = BatchNormalization()(feature)   # before dropout or after?
x = Dense(120, activation = 'linear')(x)
x = PReLU()(x)

x = BatchNormalization()(x)   # before dropout or after? before seems to train faster
#x = Dropout(0.25)(x)
x = Dense(80, activation = 'linear')(x)
x = PReLU()(x)

#x = BatchNormalization()(x)   # before dropout or after?
#x = Dropout(0.25)(x)
f = Dense(80, activation = 'linear', use_bias=False)(feature)
x = add([f, x])
x = PReLU()(x)

#----------------------------------------------------
xp = BatchNormalization()(x)
xp = Dense(30, activation = 'linear')(xp)
xp = PReLU()(xp)
xp = BatchNormalization()(xp)
xp = Dense(30, activation = 'linear')(xp)
xp = PReLU()(xp)
#xp = BatchNormalization()(xp)
#
x = Dense(30, activation = 'linear', use_bias=False)(x)
x = add([x , xp])
x = PReLU()(x)
##
##
#xp = BatchNormalization()(x)
#xp = Dense(30, activation = 'relu')(x)
#xp = BatchNormalization()(xp)
#xp = Dense(30, activation = 'relu')(xp)
##xp = BatchNormalization()(xp)
#
#x = add([x , xp])
#x = PReLU()(x)
#
#xp = Dense(30, activation = 'relu')(x)
#xp = BatchNormalization()(xp)
#
#xp = Dense(30, activation = 'relu')(xp)
#xp = BatchNormalization()(xp)
#
#x = add([x , xp])
#x = PReLU()(x)
#---------------------------------------------------
x = BatchNormalization()(x)
pred = Dense(cls, activation = 'softmax')(x)

model = Model(feature, pred)

x_encoder = Model(feature, x) # use a different symbol than x?

#best_model_file = "leafnet.h5"
#best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=0, save_best_only=True)
best_model=EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.compile(loss = losses.categorical_crossentropy,
            optimizer = optimizers.RMSprop(),
            metrics = ['accuracy'])

history = model.fit(x=x_train_std, y=y_train,
                    batch_size = batchsize,
                    epochs = epochs, verbose = 0,
                    validation_data = (x_test_std, y_test),
                    callbacks=[best_model])


#------------------------------------------------------------------------------
# Learning curve
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt

def LC(history):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


score = model.evaluate(x_test_std, y_test)
print('test loss:', score[0])
print('test accuracy:', score[1]) 

from sklearn.metrics import coverage_error
prob = model.predict(x_test_std)
print('True labels are within %.2f of the prediction' 
      % (coverage_error(y_test, prob)))

LC(history)

#plt.figure()
#for i in range(cls):
#    plt.subplot(3,5,i+1)
#    plt.plot(leaf_data[75*i])


#------------------------------------------------------------------------------
# Check prediction rank for those that are classified wrong
#------------------------------------------------------------------------------
'''
In roughly half of wrong predictions, the model has the correct label as the second candidate 
'''
def check_pred(testy, prob):
    predy = np.argmax(prob, axis = 1)
    wrong_label = np.where(predy != testy)
    print("|Should be|\t\t|pred class|\n")
    for k in wrong_label[0]:
        print("{}\t\t\t{}\n".format(testy[k], np.argsort(prob[k])[-2:]))

#------------------------------------------------------------------------------
# A different classifier
#------------------------------------------------------------------------------
different_classifier = False
if different_classifier:
    
#    x_train_1 = np.hstack((x_train, curvefft(x_train)))
#    x_test_1 = np.hstack((x_test, curvefft(x_test)))
#    
#    Scaler1 = StandardScaler().fit(x_train_1)
#    x_train_1 = Scaler1.transform(x_train_1)
#    x_test_1 = Scaler1.transform(x_test_1)
    
    ft1 = Input(shape = (x_train_std.shape[1], ))
    x = BatchNormalization()(ft1)   # before dropout or after?
    x = Dense(120, activation = 'linear')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)   # before dropout or after?
    x = Dropout(0.4)(x)
    
    x = Dense(80, activation = 'linear')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)   # before dropout or after?
    x = Dropout(0.25)(x)
    
    pred1 = Dense(cls, activation = 'softmax')(x)
    
    model1 = Model(ft1, pred1)
      
    model1.compile(loss = losses.categorical_crossentropy,
                optimizer = optimizers.Adam(),
                metrics = ['accuracy'])
    
    history1 = model1.fit(x=x_train_std, y=y_train,
                        batch_size = batchsize,
                        epochs = epochs, verbose = 0,
                        validation_data = (x_test_std, y_test))
    
    
#    x_train_2 = np.hstack((auto_corr(x_train), curvefft(x_train)))
#    x_test_2 = np.hstack((auto_corr(x_test), curvefft(x_test)))
#    
#    Scaler2 = StandardScaler().fit(x_train_2)
#    x_train_2 = Scaler2.transform(x_train_2)
#    x_test_2 = Scaler2.transform(x_test_2)
    
    ft2 = Input(shape = (x_train_std.shape[1], ))
    xx = BatchNormalization()(ft2)   # before dropout or after?
    xx = Dense(120, activation = 'linear')(xx)
    xx = PReLU()(xx)
    xx = BatchNormalization()(xx)   # before dropout or after?
    xx = Dropout(0.4)(xx)
    
    xx= Dense(80, activation = 'linear')(xx)
    xx = PReLU()(xx)
    xx = BatchNormalization()(xx)   # before dropout or after?
    xx = Dropout(0.25)(xx)
    
    pred2 = Dense(cls, activation = 'softmax')(xx)
    
    model2 = Model(ft2, pred2)
      
    model2.compile(loss = losses.categorical_crossentropy,
                optimizer = optimizers.Adam(),
                metrics = ['accuracy'])
    
    history2 = model2.fit(x=x_train_std, y=y_train,
                        batch_size = batchsize,
                        epochs = epochs, verbose = 0,
                        validation_data = (x_test_std, y_test))
    
    y_pred = 0.5*(model1.predict(x_test_std) + model2.predict(x_test_std))
    
    print("merge accuracy: %.4f" % accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis = 1)))
    


#------------------------------------------------------------------------------
# A second stage classification
#------------------------------------------------------------------------------
stg2 = False

if stg2:
    from sklearn import svm 
#    from sklearn.decomposition import PCA 
    # locate samples that have low predictive power
    
    y_pred = np.argmax(prob, axis = 1)
    
    cut = 0.5
    seats = 2
    
    stg2_ind = [] # record the test sample that needs go to stage II
    stg2_cand = [] # record top candidated class for further investigation
    for j, pin in enumerate(prob):
        if np.max(pin) < cut:
            stg2_ind.append(j)
            stg2_cand.append([np.argsort(pin)[-seats:]])
    
    
    y_train_stg2 = np.argmax(y_train, axis = 1)
    y_pred_copy = y_pred.copy()
    
    for j, topcls in enumerate(stg2_cand):   
        
        # Use SVM
        clf_stg2 = svm.SVC(C=1.1,cache_size=200, class_weight='balanced', coef0=0,
            decision_function_shape='ovr', degree=2, gamma='auto', kernel='linear',
            max_iter=-1, probability=True, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        # Use GPC?
            
        xx_train = []
        yy_train = []
        for i in range(seats):
            selected_class = np.where(y_train_stg2 == topcls[0][i])
            xx_train.append(x_train_stack[selected_class]) 
            yy_train.append(y_train_stg2[selected_class])
                
        xx_train = np.vstack(xx_train)
        yy_train = np.hstack(yy_train)
                
        #    xx_train = xx_train.reshape(xx_train.shape[0], -1)       
        xx_test = x_test_stack[stg2_ind[j]].reshape(1, -1)
            
        scaler_stg2 = StandardScaler().fit(xx_train)
        xx_train_std = scaler_stg2.transform(xx_train) 
        xx_test_std = scaler_stg2.transform(xx_test)
                    
        pre_pca = PCA(n_components=30).fit(xx_train_std) 
        xx_train_pca = pre_pca.transform(xx_train_std)
        xx_test_pca = pre_pca.transform(xx_test_std)
                
        clf_stg2.fit(xx_train_pca, yy_train)
#        prob_svm = clf_stg2.predict_proba(xx_test_pca) 
#        y_pred_copy[stg2_ind[j]] = stg2_cand[j][0][np.argmax(prob_svm + prob[stg2_ind[j], stg2_cand[j]])]
            
        y_pred_copy[stg2_ind[j]] = clf_stg2.predict(xx_test_pca)
        
    print('accruracy after stage II: %.4f' % accuracy_score(np.argmax(y_test, axis=1), y_pred_copy) )


#------------------------------------------------------------------------------
# A second nn with weights adjusted from the first nn
#------------------------------------------------------------------------------
r_mode = False
if r_mode:
    from keras.layers import multiply
    epochs_merge = 80
    feature_r = Input(shape= (input_dim, ) )
    
    y = BatchNormalization()(feature_r)   # before dropout or after?
    y = Dense(120, activation = 'relu')(y)
    y = BatchNormalization()(y)   # before dropout or after?
    y = Dropout(0.25)(y)
    
    y = Dense(80, activation = 'relu')(y)
    y = BatchNormalization()(y)   # before dropout or after?
    y = Dropout(0.25)(y)
    
#    pred_r = Dense(cls, activation = 'softmax')(y)
    
    yy = BatchNormalization()(feature_r)   # before dropout or after?
    yy = Dense(120, activation = 'relu')(yy)
    yy = BatchNormalization()(yy)   # before dropout or after?
    yy = Dropout(0.25)(yy)
    
    yy = Dense(80, activation = 'relu')(yy)
    yy = BatchNormalization()(yy)   # before dropout or after?
    yy = Dropout(0.25)(yy)
       
#    pred_rr = Dense(cls, activation = 'softmax')(yy)

#    pred_merge = multiply([pred_r, pred_rr])
    
    pred_merge = multiply([y, yy])
    pred_merge = Dense(cls, activation = 'softmax')(pred_merge)
    model_r = Model(feature_r, pred_merge)
    
    model_r.compile(loss = losses.categorical_crossentropy,
                optimizer = optimizers.Adam(),
                metrics = ['accuracy'])
    
    history_r = model_r.fit(x=x_train_std, y=y_train,
                        batch_size = batchsize,
                        epochs = epochs_merge, verbose = 0,
                        validation_data = (x_test_std, y_test))  
    
    
    prob_r = model.predict(x_test_std)
    print('accruracy with merge: %.4f' % accuracy_score(np.argmax(y_test, axis=1), 
                                                        np.argmax(prob_r, axis=1)))
    print('True labels are within %.2f of the prediction' 
      % (coverage_error(y_test, prob_r)))

#------------------------------------------------------------------------------
# Use the pretrained network to provide feature extraction and then build a
# second classifier on top of it.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Visualize the feature space with manifold learning
#------------------------------------------------------------------------------
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier
# =============================================================================
# from sklearn.manifold import TSNE, MDS, Isomap
# from mpl_toolkits.mplot3d import Axes3D
# 
# 
# features_std  = x_encoder.predict(np.vstack((x_train_std, x_test_std)))
# label = np.vstack((y_train, y_test))
# 
# plt.figure(figsize=(10,10))
# x_embedded_2d = TSNE(n_components=2).fit_transform(features_std)
# plt.scatter(x_embedded_2d[:, 0], x_embedded_2d[:, 1], c=np.argmax(label,axis=1))
# plt.colorbar() 
# 
# fig = plt.figure(figsize=(10,10))
# ax = Axes3D(fig)
# x_embedded_3d = TSNE(n_components=3).fit_transform(features_std)
# p = ax.scatter(x_embedded_3d[:, 0], x_embedded_3d[:, 1], x_embedded_3d[:,2],
#                c=np.argmax(label,axis=1))
# fig.colorbar(p)
# =============================================================================

#------------------------------------------------------------------------------
# A second classifier with pretrained features
# Use a pretrained network saved at best performance on validation set
#------------------------------------------------------------------------------

xx_train = x_encoder.predict(x_train_std)
xx_test = x_encoder.predict(x_test_std)

'''
Stacking new features at this stage? seems to help
which to stack? (a lot to tune) leaf_hist with 30 comp once get 96.65%
how many components? 1/3 ~1/2 of original dimension
'''
#xx_train = np.hstack((xx_train, leaf_hist(x_train)))
#xx_test = np.hstack((xx_test, leaf_hist(x_test)))

xx_train_std, xx_test_std = preprocess(xx_train, xx_test)
xx_train_pca, xx_test_pca = addpca(xx_train_std, xx_test_std, comp = 25)

# Using Knn for nonlinearity correction?
clf_2 = svm.SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0,
            decision_function_shape='ovr', degree=1, gamma='auto', kernel='rbf',
            max_iter=-1, probability=True, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

clf_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                                leaf_size=3, p=1, metric='minkowski', 
                                metric_params=None, n_jobs=1)

 
yy_train = np.argmax(y_train, axis = 1)

clf_2.fit(xx_train_pca, yy_train)
print("the accuracy with pretrain (svm): %.4f" % accuracy_score(np.argmax(y_test, axis=1),
                                                          clf_2.predict(xx_test_pca)))

clf_knn.fit(xx_train_std, yy_train) 
y_pred_knn = clf_knn.predict(xx_test_std)
print("the accuracy with pretrain (knn): %.4f" % accuracy_score(np.argmax(y_test, axis=1),
                                                          y_pred_knn))

'''
average voting?

Doesn't seems to be better
'''
#prob_svm = clf_2.predict_proba(xx_test_pca)
#prob_w = 0.6*prob_svm + 0.4*prob
#print("the accuracy with weights from (svm): %.4f" % accuracy_score(np.argmax(y_test, axis=1),
#                                                          np.argmax(prob_w, axis=1)))

#------------------------------------------------------------------------------
#epochs2 = 5
#fff = Input(shape= (xx_train.shape[1], ) )
#
#xxx = BatchNormalization()(fff)   # before dropout or after?
#xxx = Dense(60, activation = 'linear')(xxx)
#xxx = PReLU()(xxx)
#
#xxx = BatchNormalization()(xxx)   # before dropout or after? before seems to train faster
#xxx = Dropout(0.25)(xxx)
#
#xxx = Dense(30, activation = 'linear')(xxx)
#xxx = PReLU()(xxx)
#
#xxx = BatchNormalization()(xxx)   # before dropout or after?
#xxx = Dropout(0.2)(xxx)
#
#
#ppp = Dense(cls, activation = 'softmax')(xxx)
#
#model2 = Model(fff, ppp)
#
#xx_encoder = Model(fff, xxx)
#
#model2.compile(loss = losses.categorical_crossentropy,
#            optimizer = optimizers.Adam(),
#            metrics = ['accuracy'])
#
#history2 = model2.fit(x=xx_train, y=y_train,
#                    batch_size = batchsize,
#                    epochs = epochs2, verbose = 0,
#                    validation_data = (xx_test, y_test))
#
#score2 = model2.evaluate(xx_test, y_test)
#print('test loss after pretrain:', score2[0])
#print('test accuracy after pretrain (nn):', score2[1]) 


#------------------------------------------------------------------------------
# May improve by adding a third stage? Does not seems so...
#------------------------------------------------------------------------------

# =============================================================================
# xxx_train = xx_encoder.predict(xx_train)
# xxx_test = xx_encoder.predict(xx_test)
# 
# 
# xxx_train, xxx_test = preprocess(xxx_train, xxx_test)
# xxx_train_pca, xxx_test_pca = addpca(xxx_train, xxx_test, comp = 10)
# 
# clf_3 = svm.SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0,
#             decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear',
#             max_iter=-1, probability=True, random_state=None, shrinking=True,
#             tol=0.001, verbose=False)
# 
# clf_3.fit(xxx_train_pca, yy_train)
# print("the accuracy with pretrain: %.4f" % accuracy_score(np.argmax(y_test, axis=1),
#                                                           clf_3.predict(xxx_test_pca)))
# 
# 
# #------------------------------------------------------------------------------
# epochs3 = 6
# 
# ff_f = Input(shape= (30, ) )
# 
# xx_x = BatchNormalization()(ff_f)   # before dropout or after?
# xx_x = Dense(20, activation = 'linear')(xx_x)
# xx_x = PReLU()(xx_x)
# 
# xx_x = BatchNormalization()(xx_x)   # before dropout or after? before seems to train faster
# xx_x = Dropout(0.25)(xx_x)
# 
# xx_x = Dense(20, activation = 'linear')(xx_x)
# xx_x = PReLU()(xx_x)
# 
# xx_x = BatchNormalization()(xx_x)   # before dropout or after?
# xx_x = Dropout(0.2)(xx_x)
# 
# 
# pp_p = Dense(cls, activation = 'softmax')(xx_x)
# 
# model3 = Model(ff_f, pp_p)
# 
# 
# model3.compile(loss = losses.categorical_crossentropy,
#             optimizer = optimizers.Adam(),
#             metrics = ['accuracy'])
# 
# history3 = model3.fit(x=xxx_train, y=y_train,
#                     batch_size = batchsize,
#                     epochs = epochs3, verbose = 0,
#                     validation_data = (xxx_test, y_test))
# 
# score3 = model3.evaluate(xxx_test, y_test)
# print('test loss:', score3[0])
# print('test accuracy:', score3[1]) 
# =============================================================================



