# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:05:42 2018

@author: Dongyang

This script uses 10 fold cv to test the performance of this classifier on
the Swedish leaf dataset.
"""


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, BatchNormalization, add, Conv1D, Flatten, concatenate, MaxPooling1D, GaussianNoise
from keras.models import Model, load_model
from keras import backend as K
from keras import optimizers, losses, utils
from sklearn.preprocessing import  RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder, Normalizer, QuantileTransformer
from sklearn.metrics import accuracy_score
import numpy as np
from keras.callbacks import  ModelCheckpoint
from keras.regularizers import l2
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier
from keras.layers.advanced_activations import PReLU
#------------------------------------------------------------------------------
# Read data
#------------------------------------------------------------------------------

target_dir = 'data//leaf//'
batchsize = 32
epochs = 60

cls = 15 
size = 75 * 15
 
#leaf_data = np.zeros((size, 200))
#leaf_label = np.zeros(size)
#
#
#for i in range(cls):
#    leaf_data[i*75:(i+1)*75] = np.load(target_dir + 'S_leaf_CCD{}.npy'.format(i+1))
#    leaf_label[i*75:(i+1)*75] = i
    
leaf_data = np.load(target_dir + 'S_leaf_CCD.npy')
leaf_label = np.load(target_dir + 'S_leaf_label.npy')
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
#import csv
#target = r'data/100 leaves plant species/data_Mar_64.txt'
# 
#leaf_Mar = []
#with open(target) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         leaf_Mar.append(row)
#              
#leaf_Mar = np.asarray(leaf_Mar)
#leaf_Mar = leaf_Mar[16:,1:].astype(float)
# 
#target = r'data/100 leaves plant species/data_Sha_64.txt'
#leaf_Sha = []
#with open(target) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         leaf_Sha.append(row)         
#leaf_Sha = np.asarray(leaf_Sha)
#leaf_Sha = leaf_Sha[16:,1:].astype(float)
# 
# 
#target = r'data/100 leaves plant species/data_Tex_64.txt'
# 
#leaf_Tex = []
#with open(target) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         leaf_Tex.append(row)
#              
#leaf_Tex = np.asarray(leaf_Tex)
#leaf_Tex = leaf_Tex[15:,1:].astype(float)
# 
#leaf_label = np.zeros(1584)
# 
#for i in range(99):
#     leaf_label[16*i:16*i+16] = i
#     
#cls=99
#size = 1584
# 
#
##leaf_data = np.hstack([leaf_Sha, leaf_Tex , leaf_Mar])
#leaf_data = np.hstack([leaf_Sha, leaf_Sha , leaf_Sha])
# =============================================================================
#leaf_data = leaf_Sha
#------------------------------------------------------------------------------
# Some Util functions

def preprocess(train, test, flag = True):
    if True:
        scaler = StandardScaler().fit(train)
#        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train)
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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
Kf = 10
#skf = StratifiedKFold(n_splits=Kf, shuffle=True, random_state=333)
skf = StratifiedShuffleSplit(n_splits=Kf, test_size = 1./Kf, random_state = 0)
cv_acc_svm = np.zeros(Kf)
cv_acc_knn = np.zeros(Kf)
cv_acc = np.zeros(Kf)

i=0

for train_index, test_index in skf.split(leaf_data, leaf_label):
     x_train, x_test = leaf_data[train_index], leaf_data[test_index]
     y_train, y_test = leaf_label[train_index], leaf_label[test_index]
     
#     aug_flag = False
#     if aug_flag:
#         x_train = np.vstack((x_train, 
#                              np.flip(x_train, axis = 1)
#                              ))
#         y_train = np.hstack((y_train, y_train))
#         
#         x_train = np.vstack((np.roll(x_train, 5, axis = 1),
#                              np.roll(x_train, -5, axis = 1)))
#         y_train = np.hstack((y_train, y_train))
         
     
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
     input_dim = x_train_std.shape[1]
     feature = Input(shape = (input_dim, 1))
     
     x = GaussianNoise(0.01)(feature)
     x = Conv1D(filters= 16, kernel_size = 8, strides=4, padding='same', dilation_rate=1, 
            activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
            name = 'conv1D_1')(x)
     x = BatchNormalization()(x)
#     x = PReLU()(x)
     x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_1')(x)
#     x = Dropout(0.25)(x)
     x = Flatten(name = 'flat_1')(x)
     
     x_x = GaussianNoise(0.01)(feature)
     x_x = Conv1D(filters= 24, kernel_size = 12, strides= 6, padding='same', dilation_rate=1, 
            activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
            name = 'conv1D_2')(x_x)
     x_x = BatchNormalization()(x_x)
#     x_x = PReLU()(x_x)
     x_x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_2')(x_x)
#     x_x = Dropout(0.25)(x_x)
     x_x = Flatten()(x_x)
     
     x_x_x = GaussianNoise(0.01)(feature)
     x_x_x = Conv1D(filters= 32, kernel_size = 16, strides= 8, padding='same', dilation_rate=1, 
            activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
            name = 'conv1D_3')(x_x_x)
     x_x_x = BatchNormalization()(x_x_x)
#     x_x_x = PReLU()(x_x_x)
     x_x_x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_3')(x_x_x)
#     x_x_x = Dropout(0.25)(x_x_x)
     x_x_x = Flatten()(x_x_x)
     
     feature_f = GaussianNoise(0.01)(feature)
#     feature_f = MaxPooling1D(pool_size=4, strides=2, name = 'MP_4')(feature_f)
#     feature_f = Dropout(0.25)(feature_f)
     feature_f = Flatten(name = 'flat_2')(feature_f)
     #
     x = concatenate([x, x_x, x_x_x, feature_f])
     
     x = Dense(512, activation = 'linear', name = 'dense_1')(x)
     x = BatchNormalization()(x)
     x = PReLU()(x)
     
     x = Dense(128, activation = 'linear', name = 'dense_2')(x) #increase the dimension here for better speration in stage2 ?
     x = BatchNormalization()(x)
     x = PReLU()(x)
     
     x = Dropout(0.5)(x)
     pred = Dense(cls, activation = 'softmax', name = 'dense_3')(x)
     
     model = Model(feature, pred)

     
     #best_model=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
     best_model = ModelCheckpoint(target_dir+'leaf_conv1d_cv%d.hdf5' %i, monitor='val_loss', 
                                  verbose=0, save_best_only=True, save_weights_only=False, 
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
                         validation_split = 0.2,
#                         validation_data = (x_test_std, y_test),
                         callbacks=[best_model])
     
#     cv_acc[i] = model.evaluate(x_test_std, y_test)[1]
     #------------------------------------------------------------------------------
     # A second stage classification with features pretrained from above network
     #------------------------------------------------------------------------------
     
    
     model_best = load_model(target_dir + 'leaf_conv1d_cv%d.hdf5' %i)
     cv_acc[i] = model_best.evaluate(x_test_std, y_test)[1]
     
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
                 decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear',
                 max_iter=-1, probability=True, random_state=None, shrinking=True,
                 tol=0.001, verbose=False)
     
     clf_2.fit(xx_train_pca, yy_train)
     cv_acc_svm[i] = accuracy_score(np.argmax(y_test, axis=1), clf_2.predict(xx_test_pca))
     
     clf_knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto',
                                     leaf_size=10, p=2, metric='chebyshev', 
                                     metric_params=None, n_jobs=1)  
     
     clf_knn.fit(xx_train_std, yy_train) 
     y_pred_knn = clf_knn.predict(xx_test_std)
     cv_acc_knn[i] = accuracy_score(np.argmax(y_test, axis=1), y_pred_knn)
     
     print('the %d th validation finished....' % i)
     print('accuracy %.4f' % cv_acc[i])
     i+=1
                                                               
