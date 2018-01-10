# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:40:01 2017

@author: dykua

This script tries to use transfer learning on top of VGG19 to do the leaf 
classification
"""

import numpy as np
from sklearn.model_selection import train_test_split
#from keras import applications
from keras import utils, losses
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model 
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense, MaxPooling2D
#from keras import backend as k 
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width, img_height = 256, 256
size = 340
cls = 30
batch_size = 128
epochs = 5


target_dir = 'data//leaf//'
leaf_data = np.load(target_dir+'leaf_data_{}_{}.npy'.format(img_height, img_width))
leaf_label = np.load(target_dir+'leaf_label_32_24.npy')

#leaf_data = leaf_data.reshape(size, img_height, img_width, 1)

leaf_data_rgb = np.empty((size, img_height, img_width, 3)) # adding different features to the rest channels?
leaf_data_rgb [:, :, :, 0] = leaf_data
leaf_data_rgb [:, :, :, 1] = leaf_data
leaf_data_rgb [:, :, :, 2] = leaf_data

x_train, x_test, y_train, y_test = train_test_split(
                             leaf_data_rgb - np.mean(leaf_data_rgb, axis = 0), leaf_label-1, 
                             test_size=0.10, #careful here
                             random_state=42,
                             shuffle = True)

y_train = utils.to_categorical(y_train, cls)
y_test = utils.to_categorical(y_test, cls)

### Build the network 
img_input = Input(shape=(256, 256, 3))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Add on top a dense layer
x = MaxPooling2D((20, 20), strides=(20, 20), name='pool2ft')(x) # contral the size of features
h = Flatten()(x)
hh = Dropout(0.80)(h)
prediction = Dense(cls, activation = 'softmax')(hh)

model = Model(img_input, prediction)
feature_extractor = Model(img_input, h)



#model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 256, 256, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 256, 256, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 256, 256, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 128, 128, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 128, 128, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 128, 128, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 64, 64, 128)       0         
=================================================================
Total params: 260,160.0
Trainable params: 260,160.0
Non-trainable params: 0.0
"""

layer_dict = dict([(layer.name, layer) for layer in model.layers[1:7]])


import h5py
weights_path = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5' # ('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5)
f = h5py.File(weights_path)


# list all the layer names which are in the model.
layer_names = [layer.name for layer in model.layers[1:7]]



for i in layer_dict.keys():
    weight_names = f[i].attrs['weight_names']
    weights = [f[i][j] for j in weight_names]
    index = layer_names.index(i)
    model.layers[index+1].set_weights(weights)

for layer in model.layers[:7]:
   layer.trainable = False
    

#datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True,
#    zoom_range=0.1,
#    rotation_range=40,
#    width_shift_range=0.1,
#    height_shift_range=0.1,
#    horizontal_flip=True,
#    vertical_flip=True)
#
#
#datagen.fit(x_train)

feature = feature_extractor.predict(leaf_data_rgb)

#model.compile(loss = losses.categorical_crossentropy,
#            optimizer = optimizers.Adam(),       # change this to sgd        
#            metrics = ['accuracy'])   
#
#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
#                    steps_per_epoch=len(x_train) / 128, epochs = epochs,
#                    verbose=0, validation_data= datagen.flow(x_test, y_test, batch_size=32),
#                    validation_steps=100)

# list all data in history
#print(history.history.keys())

#import matplotlib.pyplot as plt
## summarize history for accuracy
#plt.figure()
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.figure()
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
#
#
#score = model.evaluate_generator(datagen.flow(x_test, y_test, batch_size=300), steps=1)
#print('test loss:', score[0])
#print('test accuracy:', score[1])  
        


