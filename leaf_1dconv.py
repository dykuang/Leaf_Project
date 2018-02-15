# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 14:25:27 2018

@author: dykuang

This script tests 1d conv net for CCD


Comments:
    It does bring noticible improvement. A sliding window that looks at local de
    details. (Check network in networks, the sliding window can be a network as 
    well for adding nonlinearity). At least, better for next stage data 
    preparation. I got a feeling that it can perform better after some
    hyper-parameter tuning. Number of parameters contained are much larger than
    my previous simple network.
    
TODO: * A data generator on the fly? (Check Keras's document about sequential model)
        or google it.(check the post with medium.com)
      * save the 'best' during training and reload it for stage two. (Done)
"""

from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, BatchNormalization, add, Conv1D, Flatten, concatenate, MaxPooling1D, GaussianNoise
from keras.models import Model, load_model
from keras import backend as K
from keras import optimizers, losses, utils
from sklearn.preprocessing import  RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder, Normalizer, QuantileTransformer
from sklearn.metrics import accuracy_score
import numpy as np
import random
from keras.callbacks import  EarlyStopping, ModelCheckpoint
from keras import regularizers

#------------------------------------------------------------------------------
# Read data
#------------------------------------------------------------------------------

target_dir = 'data//leaf//'
batchsize = 32
epochs = 60

cls = 15 
size = 75 * 15
 
leaf_data = np.zeros((size, 200))
leaf_label = np.zeros(size)


for i in range(cls):
    leaf_data[i*75:(i+1)*75] = np.load(target_dir + 'S_leaf_CCD{}.npy'.format(i+1))
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
# leaf_Mar = leaf_Mar[16:,1:].astype(float)
# 
# target = r'data/100 leaves plant species/data_Sha_64.txt'
# leaf_Sha = []
# with open(target) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         leaf_Sha.append(row)         
# leaf_Sha = np.asarray(leaf_Sha)
# leaf_Sha = leaf_Sha[16:,1:].astype(float)
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
# leaf_Tex = leaf_Tex[15:,1:].astype(float)
# 
# leaf_label = np.zeros([1584])
# 
# for i in range(99):
#     leaf_label[16*i:16*i+16] = i
#     
# cls=99
# size = 1584
# 
# leaf_data = np.hstack([leaf_Sha, leaf_Sha , leaf_Sha])
# =============================================================================
#leaf_data = leaf_Sha
#------------------------------------------------------------------------------
# Some Util functions
#----------------------------------------------------------------------------
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

import pandas as pd
def auto_corr(data, lag = 2):
    N = int(data.shape[1]/lag) - 1
    ac = np.zeros((data.shape[0], N))
    for i, ccd in enumerate(data):
        for j in range(N):
            ac[i,j] = pd.Series(ccd - np.mean(ccd)).autocorr((j+1)*lag)
    return ac

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
#------------------------------------------------------------------------------



x_train, x_test, y_train, y_test, ind_train, ind_test = train_test_split(
                             leaf_data, leaf_label, np.arange(size), 
                             test_size=0.1, 
                             random_state = 233,
                             shuffle = True, stratify = leaf_label)
'''
TODO: need to think about how to augment the data properly?
'''
aug_flag = True
if aug_flag:
    x_train = np.vstack((x_train, 
                         np.flip(x_train, axis = 1),
                         np.roll(x_train, 5, axis = 1),
                         np.roll(x_train, -5, axis = 1)
                         ))
    
    y_train = np.hstack((y_train, y_train, 
                         y_train, y_train))
    
    
#    x_test = np.vstack((x_test, 
#                         np.flip(x_test, axis = 1),
#                         np.roll(x_test, 5, axis = 1),
#                         np.roll(x_test, -5, axis = 1)
#                         ))
#    
#    y_test = np.hstack((y_test, y_test, 
#                         y_test, y_test))
#def data_gen(X, y):
    
    
    
#x_train = (x_train - np.mean(x_train, axis=1).reshape(len(x_train),1))/np.max(x_train, axis = 1). reshape(len(x_train),1)   
#x_test = (x_test - np.mean(x_test, axis=1).reshape(len(x_test),1))/np.max(x_test, axis = 1). reshape(len(x_test),1)

#x_train_stack= np.hstack((x_train,
#                     curvefft(x_train)[:,:], 
#                     auto_corr(x_train,2)
#
#                     ))
#
#x_test_stack= np.hstack((x_test,
#                    curvefft(x_test)[:,:], 
#                    auto_corr(x_test,2)
#
#                    ))

#
x_train_stack = x_train
x_test_stack = x_test

y_train = utils.to_categorical(y_train, cls)
y_test = utils.to_categorical(y_test, cls)

# normalization
#scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train_stack)
#scaler = QuantileTransformer().fit(x_train_stack)
scaler = StandardScaler().fit(x_train_stack)
x_train_std = scaler.transform(x_train_stack) 
x_test_std = scaler.transform(x_test_stack)


#------------------------------------------------------------------------------
# Model 
#------------------------------------------------------------------------------
'''
Comments:
     The training accuracy cannot get close to 1.0, which will affects the test 
     accuracy.
'''

from keras.layers.advanced_activations import PReLU
input_dim = x_train_std.shape[1]
feature = Input(shape = (input_dim, 1))

x = GaussianNoise(0.02)(feature)
x = Conv1D(filters= 16, kernel_size = 8, strides=4, padding='same', dilation_rate=1, 
       activation='linear', use_bias=True, kernel_initializer='glorot_uniform',
       bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
       activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
       name = 'conv1D_1')(x)
#x = BatchNormalization()(x)
#x = PReLU()(x)
x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_1')(x)
x = Flatten(name = 'flat_1')(x)

x_x = GaussianNoise(0.02)(feature)
x_x = Conv1D(filters= 24, kernel_size = 12, strides= 6, padding='same', dilation_rate=1, 
       activation='linear', use_bias=True, kernel_initializer='glorot_uniform',
       bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
       activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
       name = 'conv1D_2')(x_x)
#x_x = BatchNormalization()(x_x)
#x_x = PReLU()(x_x)
x_x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_2')(x_x)
x_x = Flatten()(x_x)

x_x_x = GaussianNoise(0.02)(feature)
x_x_x = Conv1D(filters= 32, kernel_size = 16, strides= 8, padding='same', dilation_rate=1, 
       activation='linear', use_bias=True, kernel_initializer='glorot_uniform',
       bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
       activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
       name = 'conv1D_3')(x_x_x)
#x_x_x = BatchNormalization()(x_x_x)
#x_x_x = PReLU()(x_x_x)
x_x_x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_3')(x_x_x)
x_x_x = Flatten()(x_x_x)


#feature_f = Conv1D(filters= 8, kernel_size = 2, strides= 2, padding='same', dilation_rate=1, 
#       activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
#       bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
#       activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
#       name = 'conv1D_4')(feature)
#feature_f = MaxPooling1D(pool_size=2, strides=2, name = 'MP_4')(feature)
feature_f = Flatten(name = 'flat_2')(feature)
#
x = concatenate([x, x_x, x_x_x, feature_f])

#x = BatchNormalization()(x) 
x = Dense(512, activation = 'linear', name = 'dense_1')(x)
x = PReLU()(x)
x = BatchNormalization()(x)


x = Dense(128, activation = 'linear', name = 'dense_2')(x) #increase the dimension here for better speration in stage2 ?
x = PReLU()(x)
x = BatchNormalization()(x)


#x = Dense(64, activation = 'linear', name = 'dense_extra')(x)
#x = PReLU()(x)
#x = BatchNormalization()(x) 

#
#x = Dropout(0.15)(x)
pred = Dense(cls, activation = 'softmax', name = 'dense_3')(x)

model = Model(feature, pred)

#best_model=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
best_model = ModelCheckpoint(target_dir+'leaf_conv1d.hdf5', monitor='val_acc', 
                             verbose=1, save_best_only=True, save_weights_only=False, 
                             mode='auto', period=1)

model.compile(loss = losses.categorical_crossentropy,
#            optimizer = optimizers.Adam(),
            optimizer = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True),
            metrics = ['accuracy'])

x_train_std = x_train_std.reshape(x_train_std.shape[0], x_train_std.shape[1], 1)
x_test_std = x_test_std.reshape(x_test_std.shape[0], x_test_std.shape[1], 1)

history = model.fit(x=x_train_std, y=y_train,
                    batch_size = batchsize,
                    epochs = epochs, verbose = 0,
                    validation_split = 0.1,
#                    validation_data = (x_test_std, y_test),
                    callbacks=[best_model])


#------------------------------------------------------------------------------
# A slightly different model 
#------------------------------------------------------------------------------

# =============================================================================
# input_dim = 200
# feature = Input(shape = (input_dim, 1))
# 
# x = Conv1D(filters= 64, kernel_size = 20, strides=10, padding='same', dilation_rate=1, 
#        activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
#        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
#        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(feature)
# 
# x = MaxPooling1D(pool_size=4, strides=2)(x)
# x = Flatten()(x)
# 
# #feature_f = Flatten()(feature)
# 
# feature_extracted = Input(shape=(199, ))
# 
# x = concatenate([x, feature_extracted])
# 
# x = BatchNormalization()(x)   
# x = Dense(300, activation = 'relu')(x)
# 
# 
# x = BatchNormalization()(x)   
# x = Dense(200, activation = 'relu')(x) 
# 
# 
# x = BatchNormalization()(x) 
# #x = Dropout(0.15)(x)
# pred = Dense(cls, activation = 'softmax')(x)
# 
# model = Model([feature, feature_extracted], pred)
# 
# x_encoder = Model([feature, feature_extracted], x) # Can use K.function to extract
# 
# best_model=EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
# model.compile(loss = losses.categorical_crossentropy,
#             optimizer = optimizers.RMSprop(),
#             metrics = ['accuracy'])
# 
# 
# ccd_train = np.expand_dims(x_train_std[:,:200], axis=3)
# train_extracted = x_train_std[:,200:]
# 
# ccd_test = np.expand_dims(x_test_std[:,:200], axis=3)
# test_extracted = x_test_std[:,200:]
# 
# history = model.fit(x=[ccd_train, train_extracted], y=y_train,
#                     batch_size = batchsize,
#                     epochs = epochs, verbose = 0,
#                     validation_data = ([ccd_test, test_extracted], y_test),
#                     callbacks=[best_model])
# =============================================================================


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

#------------------------------------------------------------------------------
# A second stage classification with features pretrained from above network
#------------------------------------------------------------------------------

from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier

model_best = load_model(target_dir + 'leaf_conv1d.hdf5')
x_encoder = K.function([model_best.layers[0].input, K.learning_phase()],
                        [model_best.get_layer('dense_3').input])

yy_train = np.argmax(y_train, axis = 1)
xx_train = x_encoder([x_train_std, 0])[0]
xx_test = x_encoder([x_test_std, 0])[0]

xx_train_std, xx_test_std = preprocess(xx_train, xx_test)
#xx_train_std, xx_test_std = xx_train, xx_test
xx_train_pca, xx_test_pca = addpca(xx_train_std, xx_test_std, comp = 25)
 
# Using Knn for nonlinearity correction?
clf_2 = svm.SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0,
            decision_function_shape='ovr', degree=1, gamma='auto', kernel='rbf',
            max_iter=-1, probability=True, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

clf_2.fit(xx_train_pca, yy_train)
print("the accuracy with pretrain (svm): %.4f" % accuracy_score(np.argmax(y_test, axis=1),
                                                          clf_2.predict(xx_test_pca)))

clf_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                                leaf_size=10, p=2, metric='chebyshev', 
                                metric_params=None, n_jobs=1)



clf_knn.fit(xx_train_std, yy_train) 
y_pred_knn = clf_knn.predict(xx_test_std)
print("the accuracy with pretrain (knn): %.4f" % accuracy_score(np.argmax(y_test, axis=1),
                                                          y_pred_knn))

#==============================================================================
# Visualization by imbedding into 2d/3d
#==============================================================================

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def view_embeded(data, label):
     plt.figure(figsize=(10,10))
     x_embedded_2d = TSNE(n_components=2).fit_transform(data)
     plt.scatter(x_embedded_2d[:, 0], x_embedded_2d[:, 1], 25, c=label, cmap = 'rainbow')
     plt.colorbar()
     
#     fig = plt.figure(figsize=(10,10))
#     ax = Axes3D(fig)
#     x_embedded_3d = TSNE(n_components=3).fit_transform(data)
#     p = ax.scatter(x_embedded_3d[:, 0], x_embedded_3d[:, 1], x_embedded_3d[:,2],
#                    25, c=label)
#     fig.colorbar(p)
#
#view_embeded(np.vstack((xx_train, xx_test)), 
#             np.hstack((yy_train, np.argmax(y_test, axis=1))))