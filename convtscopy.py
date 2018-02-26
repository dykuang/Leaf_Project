# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:55:29 2018

@author: Dongyang

This script tests if the architecture from leaf data can also

apply to some other time series.

A copy...
"""

from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, BatchNormalization, add, Conv1D, Flatten, concatenate, MaxPooling1D, GaussianNoise, AveragePooling1D
from keras.models import Model, load_model
from keras import backend as K
from keras import optimizers, losses, utils
from sklearn.preprocessing import  RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder, Normalizer, QuantileTransformer
from sklearn.metrics import accuracy_score
import numpy as np
import random
from keras.callbacks import  EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

#random.seed(1234)
#------------------------------------------------------------------------------
# Read data
#------------------------------------------------------------------------------

batchsize = 32
epochs = 60

data_train = ['datasets/Worms/Worms_TEST.txt',
              'datasets/SwedishLeaf/SwedishLeaf_TRAIN.txt',
              'datasets/Phoneme/Phoneme_TRAIN.txt',
              'datasets/CC/ChlorineConcentration_TRAIN.txt',
              'datasets/InsectWingBeatSound/InsectWingBeatSound_TRAIN.txt',
              'datasets/WordsSynonyms/WordsSynonyms_TRAIN.txt',
              'datasets/FIftyWords/50words_TRAIN.txt',
              'datasets/ElectricDevices/ElectricDevices_TRAIN.txt',
              'datasets/DistalPhalanXTW/DistalPhalanXTW_TRAIN.txt'
              ]

data_test = ['datasets/Worms/Worms_TRAIN.txt',
              'datasets/SwedishLeaf/SwedishLeaf_TEST.txt',
              'datasets/Phoneme/Phoneme_TEST.txt',
              'datasets/CC/ChlorineConcentration_TEST.txt',
              'datasets/InsectWingBeatSound/InsectWingBeatSound_TEST.txt',
              'datasets/WordsSynonyms/WordsSynonyms_TEST.txt',
              'datasets/FIftyWords/50words_TEST.txt',
              'datasets/ElectricDevices/ElectricDevices_TEST.txt',
              'datasets/DistalPhalanXTW/DistalPhalanXTW_TEST.txt']

save_dir = ['datasets/Worms/',
            'datasets/SwedishLeaf/',
            'datasets/Phoneme/',
            'datasets/CC/',
             'datasets/InsectWingBeatSound/',
             'datasets/WordsSynonyms/',
             'datasets/FiftyWords/',
             'datasets/ElectricDevices/',
             'datasets/DistalPhalanXTW/']


clsss=[5, 15, 39, 3, 11, 25, 50, 7, 6]

acc_reported = [0.7349, 0.9667, 0.3620, 0.8457, 0.6389, 0.7784, 0.8207, 0.8954, 0.6932]
sig_len = [900,128,1024,166, 256, 270, 270, 96, 80]

flag = 7
import csv
target_train = data_train[flag]
target_test = data_test[flag]
cls = clsss[flag]
target_dir = save_dir[flag]

xtrain = []
with open(target_train) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        xtrain.append(row)
             
xtrain = np.asarray(xtrain)
y_train = xtrain[:,0].astype(float)-1
x_train = xtrain[:,1:].astype(float)

xtest = []
with open(target_test) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        xtest.append(row)
             
xtest = np.asarray(xtest)
y_test = xtest[:,0].astype(float)-1
x_test = xtest[:,1:].astype(float)


if flag == 8:
    y_train, y_test = y_train-2, y_test -2
    



#------------------------------------------------------------------------------
# Some Util functions
#----------------------------------------------------------------------------

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

#scaler = MinMaxScaler(feature_range=(0, 1)).fit(xtrain)
##scaler = QuantileTransformer().fit(x_train_stack)
scaler = StandardScaler().fit(xtrain)
x_train_std = scaler.transform(xtrain) 
x_test_std = scaler.transform(xtest)

#x_train_std = xtrain
#x_test_std = xtest

y_train = utils.to_categorical(y_train, cls)
y_test = utils.to_categorical(y_test, cls)
#------------------------------------------------------------------------------
# Model 
#------------------------------------------------------------------------------
'''
Comments:
     The training accuracy cannot get close to 1.0, which will affects the test 
     accuracy.
'''
from keras.layers.advanced_activations import PReLU
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier
acc = np.zeros(5)
acc_svm = np.zeros(5)
acc_knn= np.zeros(5)
for j in range(5):
    input_dim = x_train_std.shape[1]
    feature = Input(shape = (input_dim, 1))
    
    x = GaussianNoise(0.01)(feature)
    x = Conv1D(filters= 16, kernel_size = 8, strides=4, padding='same', dilation_rate=1, 
           activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
           activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
           name = 'conv1D_1')(x)
    x = BatchNormalization()(x)
    #x = PReLU()(x)
    x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_1')(x)
    x = Flatten(name = 'flat_1')(x)
    
    x_x = GaussianNoise(0.01)(feature)
    x_x = Conv1D(filters= 24, kernel_size = 12, strides= 6, padding='same', dilation_rate=1, 
           activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
           activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
           name = 'conv1D_2')(x_x)
    x_x = BatchNormalization()(x_x)
    #x_x = PReLU()(x_x)
    x_x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_2')(x_x)
    x_x = Flatten()(x_x)
    
    x_x_x = GaussianNoise(0.01)(feature)
    x_x_x = Conv1D(filters= 32, kernel_size = 16, strides= 8, padding='same', dilation_rate=1, 
           activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
           activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
           name = 'conv1D_3')(x_x_x)
    x_x_x = BatchNormalization()(x_x_x)
    #x_x_x = PReLU()(x_x_x)
    x_x_x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_3')(x_x_x)
    x_x_x = Flatten()(x_x_x)
    
    
    feature_f = GaussianNoise(0.01)(feature)
    #feature_f = MaxPooling1D(pool_size=2, strides=2, name = 'MP_4')(feature_f)
    feature_f = Flatten(name = 'flat_2')(feature_f)
    #
    x = concatenate([x, x_x, x_x_x, feature_f])
    #x = BatchNormalization()(x) 
    #x = Dropout(0.5)(x)
    
    x = Dense(512, activation = 'linear', name = 'dense_1')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Dropout(0.5)(x)
    
    x = Dense(128, activation = 'linear', name = 'dense_2')(x) #increase the dimension here for better speration in stage2 ?
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    
    
    pred = Dense(cls, activation = 'softmax', name = 'dense_3')(x)
    
    model = Model(feature, pred)
    
    #best_model=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    best_model = ModelCheckpoint(target_dir+'best_model.hdf5', monitor='val_acc', 
                                 verbose=0, save_best_only=True, save_weights_only=False, 
                                 mode='auto', period=1)
    
    model.compile(loss = losses.categorical_crossentropy,
                optimizer = optimizers.Adam(),
    #            optimizer = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True),
                metrics = ['accuracy'])
    
    x_train_std = x_train_std.reshape(x_train_std.shape[0], x_train_std.shape[1], 1)
    x_test_std = x_test_std.reshape(x_test_std.shape[0], x_test_std.shape[1], 1)
    
    history = model.fit(x=x_train_std, y=y_train,
                        batch_size = batchsize,
                        epochs = epochs, verbose = 0,
                        validation_split = 0.20,
#                        validation_data = (x_test_std, y_test),
                        callbacks=[best_model])
#

#------------------------------------------------------------------------------
# Learning curve
#------------------------------------------------------------------------------
#import matplotlib.pyplot as plt
#
#def LC(history):
#    plt.figure()
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()
#    # summarize history for loss
#    plt.figure()
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()
#
#
#score = model.evaluate(x_test_std, y_test)
#print('test loss:', score[0])
#print('test accuracy:', score[1]) 
#
#from sklearn.metrics import coverage_error
#prob = model.predict(x_test_std)
#print('True labels are within %.2f of the prediction' 
#      % (coverage_error(y_test, prob)))

#LC(history)

#------------------------------------------------------------------------------
# A second stage classification with features pretrained from above network
#------------------------------------------------------------------------------
    model_best = load_model(target_dir + 'best_model.hdf5')
    x_encoder = K.function([model_best.layers[0].input, K.learning_phase()],
                            [model_best.get_layer('dense_3').input])
    acc[j] = model_best.evaluate(x_test_std, y_test)[1]
    yy_train = np.argmax(y_train, axis = 1)
    xx_train = x_encoder([x_train_std, 0])[0]
    xx_test = x_encoder([x_test_std, 0])[0]
    
    
    #xx_train =model.predict(x_train_std)
    #xx_test = model.predict(x_test_std)
    xx_train_std, xx_test_std = preprocess(xx_train, xx_test)
    #xx_train_std, xx_test_std = xx_train, xx_test
    xx_train_pca, xx_test_pca = addpca(xx_train_std, xx_test_std, comp = 25)
     
    # Using Knn for nonlinearity correction?
    clf_2 = svm.SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0,
                decision_function_shape='ovr', degree=1, gamma='auto', kernel='rbf',
                max_iter=-1, probability=True, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
    
    clf_2.fit(xx_train_pca, yy_train)
#    print("the accuracy with pretrain (svm): %.4f" % accuracy_score(np.argmax(y_test, axis=1),
#                                                              clf_2.predict(xx_test_pca)))
    acc_svm[j] = accuracy_score(np.argmax(y_test, axis=1), clf_2.predict(xx_test_pca))
    clf_knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto',
                                    leaf_size=10, p=2, metric='chebyshev', 
                                    metric_params=None, n_jobs=1)
    
    
    
    clf_knn.fit(xx_train_std, yy_train) 
    y_pred_knn = clf_knn.predict(xx_test_std)
#    print("the accuracy with pretrain (knn): %.4f" % accuracy_score(np.argmax(y_test, axis=1),
#                                                              y_pred_knn))
#
    acc_knn[j] = accuracy_score(np.argmax(y_test, axis=1), y_pred_knn)
#==============================================================================
# Visualization by imbedding into 2d/3d
#==============================================================================

#from sklearn.manifold import TSNE
#from mpl_toolkits.mplot3d import Axes3D
#
#
#def view_embeded(data, label):
#     plt.figure(figsize=(10,10))
#     x_embedded_2d = TSNE(n_components=2).fit_transform(data)
#     plt.scatter(x_embedded_2d[:, 0], x_embedded_2d[:, 1], 25, c=label, cmap = 'rainbow')
#     plt.colorbar()
     
#     fig = plt.figure(figsize=(10,10))
#     ax = Axes3D(fig)
#     x_embedded_3d = TSNE(n_components=3).fit_transform(data)
#     p = ax.scatter(x_embedded_3d[:, 0], x_embedded_3d[:, 1], x_embedded_3d[:,2],
#                    25, c=label)
#     fig.colorbar(p)
#
#view_embeded(np.vstack((xx_train, xx_test)), 
#             np.hstack((yy_train, np.argmax(y_test, axis=1))))


