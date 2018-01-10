# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 20:24:30 2017

@author: dykua

This script uses pnn to classify leaves based on different features


CCD: Top 1 performance ~64%
CCD + FFT(no blackman window): Top 1 performance ~70.5%
"""

from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout
from keras.models import Model
#from keras import backend as K
from keras import optimizers, losses, utils
from sklearn.preprocessing import  RobustScaler
#from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


img_height, img_width = 480, 360
cls = 30
size = 340
batchsize = 128
epochs = 500

target_dir = 'data//leaf//'
leaf_data = np.load(target_dir+'leaf_data_{}_{}_CCD.npy'.format(img_height, img_width))
#leaf_data = np.load(target_dir+'leaf_data_vgg19.npy')
leaf_label = np.load(target_dir+'leaf_label_{}_{}.npy'.format(img_height, img_width))

#------------------------------------------------------------------------------
# fft
#------------------------------------------------------------------------------
from scipy.fftpack import fft
from scipy.signal import blackman

N = leaf_data.shape[1]
w = blackman(N)

leaf_fft = np.zeros([len(leaf_data), N//2])
for i, leaf in enumerate(leaf_data):
   leaf = fft(leaf)
   leaf_fft[i] = 2*np.abs(leaf[:N//2])/N
           
#leaf_fft = leaf_fft[:, :6]
#leaf_data = ((leaf_data.transpose() - np.mean(leaf_data, axis = 1))/np.max(leaf_data, axis = 1))

#leaf_data = leaf_data.reshape(-1, 1152)
leaf_data = np.hstack((leaf_data, leaf_fft))
x_train, x_test, y_train, y_test = train_test_split(
                             leaf_data, leaf_label-1, test_size=0.10, #careful here
                             random_state=42,
                             shuffle = True)

y_train = utils.to_categorical(y_train, cls)
y_test = utils.to_categorical(y_test, cls)

#------------------------------------------------------------------------------
#Preprocess: normalization, pca, kernel pca?
#------------------------------------------------------------------------------

scaler = RobustScaler().fit(x_train)
x_train_std = scaler.transform(x_train) 
x_test_std = scaler.transform(x_test)

#x_train_std = x_train_std.reshape

input_dim = x_train_std.shape[1]

#x_train_std= np.expand_dims(x_train_std, axis=0)
#x_test_std= np.expand_dims(x_test_std, axis=0)

feature = Input(shape= (input_dim, ) )
x = Dense(50, activation = 'relu')(feature)
x = Dropout(0.5)(x)
x = Dense(30, activation = 'relu')(x)
x = Dropout(0.25)(x)
#x = Dense(30, activation = 'relu')(x)
pred = Dense(cls, activation = 'softmax')(x)

model = Model(feature, pred)

#model = Sequential()
#model.add(Dense(300, input_shape = (1152, ), activation = 'relu'))
#model.add(Dense(100, activation = 'relu'))
#model.add(Dense(cls, activation = 'softmax'))

model.compile(loss = losses.categorical_crossentropy,
            optimizer = optimizers.Adam(),
            metrics = ['accuracy'])

history = model.fit(x=x_train_std, y=y_train,
                    batch_size = batchsize,
                    epochs = epochs, verbose = 0,
                    validation_data = (x_test_std, y_test))


import matplotlib.pyplot as plt
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









