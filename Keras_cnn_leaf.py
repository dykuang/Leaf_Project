# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:06:32 2017

@author: dykua

This script trains a CNN with augmented leaf data shape
"""
from __future__ import print_function

import numpy as np
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
#from keras import regularizers, initializers


'''
Load data, split it  into train/test set
'''
target_dir = 'data//leaf//'
leaf_data = np.load(target_dir+'leaf_data_64_48.npy')
leaf_label = np.load(target_dir+'leaf_label_64_48.npy')


img_height, img_width = 64, 48
cls = 30
size = 340

leaf_data = leaf_data.reshape(size, img_height, img_width, 1)
x_train, x_test, y_train, y_test = train_test_split(
                             leaf_data - np.mean(leaf_data, axis = 0), leaf_label-1, 
                             test_size=0.10, #careful here
                             random_state=42,
                             shuffle = True)

y_train = keras.utils.to_categorical(y_train, cls)
y_test = keras.utils.to_categorical(y_test, cls)
#print(x_train.shape)


'''
Build convnet
'''
input_shape = (img_height, img_width, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3),
                 activation = 'relu',
                 input_shape = input_shape)) 
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))
model.add(Flatten())
#model.add(Dense(256, activation = 'relu', activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(cls, activation = 'softmax'))


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    zoom_range=0.1,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)


datagen.fit(x_train)

#datagen.fit(x_test)

#validation_generator = datagen
#validation_generator.fit(x_train)
# fits the model on batches with real-time data augmentation:

model.compile(loss = keras.losses.categorical_crossentropy,
            optimizer = keras.optimizers.Adam(),
            metrics = ['accuracy'])   

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                    steps_per_epoch=len(x_train) / 128, epochs = 200,
                    verbose=0, validation_data= datagen.flow(x_test, y_test, batch_size=32),
                    validation_steps=100)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
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


score = model.evaluate_generator(datagen.flow(x_test, y_test, batch_size=300), steps=1)
print('test loss:', score[0])
print('test accuracy:', score[1])  

#==============================================================================
# '''
# Preview some augmented image
# '''
# from matplotlib import pyplot
# for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
# 	# create a grid of 3x3 images
# 	for i in range(0, 9):
# 		pyplot.subplot(330 + 1 + i)
# 		pyplot.imshow(X_batch[i].reshape(64, 48), cmap=pyplot.get_cmap('gray'))
# 	# show the plot
# 	pyplot.show()
# 	break
#==============================================================================
