# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:14:59 2017

@author: dykua

This script tests the idea that uses an encoder training on a 
single class to extract features. 

Classification is done on encoder level or compute the distance from the 
generated average.
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import norm

#from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Lambda, Layer, Flatten
from keras.models import Model
from keras import backend as K
from keras import metrics
#from keras.preprocessing.image import ImageDataGenerator
#from keras.datasets import mnist

img_height, img_width = 64, 48
cls = 30
size = 340

target_dir = 'data//leaf//'
leaf_data = np.load(target_dir+'leaf_data_{}_{}.npy'.format(img_height, img_width))
leaf_label = np.load(target_dir+'leaf_label_{}_{}.npy'.format(img_height, img_width))

batch_size = 128
original_dim = img_height*img_width
latent_dim = 18    # if latent dimension is larger than 2, use a t-SNE to visualize
intermediate_dim = 256
epochs = 400
epsilon_std = 1




x = Input(shape=(original_dim, ))
#x = Flatten()(x)
h = Dense(intermediate_dim, activation='relu')(x)
h = Dense(64, activation = 'relu')(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# need to modify
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
#    epsilon = 1
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h0 = Dense(64, activation='relu')
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')

h1 = decoder_h0(z)
h_decoded = decoder_h(h1)
x_decoded_mean = decoder_mean(h_decoded)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x,x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#        tangent_loss = K.sum(K.square(z_log_var), axis = -1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
encoder = Model(x, z_mean)

decoder_input = Input(shape=(latent_dim,))
_h0_decoded = decoder_h0(decoder_input)
_h_decoded = decoder_h(_h0_decoded)
_x_decoded_mean = decoder_mean(_h_decoded)

generator = Model(decoder_input, _x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=None) # this optimizer seems to work best


# train the VAE on leaf data
leaf_data = leaf_data.reshape(size, img_height, img_width, 1)
#x_train, x_test, y_train, y_test = train_test_split(
#                             leaf_data, leaf_label-1, test_size=0.10, #careful here
#                             random_state=42,
#                             shuffle = True)


train_list = []
test_list = []

test_size_per_cls = int(4)
test_size_total = test_size_per_cls*cls
for i in range(cls):
    cls_ind = np.where(leaf_label==i+1)[0]
    train_list.append(cls_ind[:-test_size_per_cls])
    test_list.append(cls_ind[-test_size_per_cls:])





AVE_encoder = np.zeros([len(train_list), latent_dim])
AVE_original = np.zeros([len(train_list), img_height*img_width])
AVE_img = np.zeros([len(train_list), img_height, img_width])
dist = np.ones([cls, test_size_total ])
dist_raw = np.ones([cls, test_size_total ])
dist_encoded = np.ones([cls, test_size_total])

for i, ind in enumerate(train_list):
    x_train = leaf_data[ind]

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))


    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
#            validation_data=(x_test, None),
            verbose = 0)   # Does it reinitialize every time?
    
    AVE_encoder[i]= np.mean(encoder.predict(x_train), axis = 0)
    AVE_original[i] = np.mean(x_train, axis = 0)
    ave = generator.predict(AVE_encoder[i].reshape([1,latent_dim]))
    AVE_img[i] = ave.reshape(img_height, img_width)


#------------------------------------------------------------------------------
# Show a comparison between one sample and the reproduced average
# An alignment before training may give better result
#------------------------------------------------------------------------------

def visualize_average(which_cls):
    
    plt.figure(figsize=(10,3))
    for j in range(5):
        plt.subplot(3,5, 1 + j )
        plt.imshow(AVE_img[j+which_cls])
        plt.subplot(3,5 ,6 + j )
        plt.imshow(leaf_data[train_list[j+which_cls][0], :, :, 0])  
        plt.subplot(3,5, 11 + j )
        plt.imshow(AVE_original[j+which_cls].reshape([img_height, img_width]))    
    
x_test = leaf_data[np.array(test_list).flatten()]
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_test_encoded = encoder.predict(x_test)
for i in range(cls):
    for j in range(test_size_total):
        dist[i, j] = np.linalg.norm(x_test[j] - AVE_img[i].flatten()) # try dist after affine transformation
        dist_raw[i, j] = np.linalg.norm(x_test[j] - AVE_original[i].flatten())
        dist_encoded[i,j] = np.linalg.norm(x_test_encoded[j] - AVE_encoder[i].flatten())
        
pred_img = np.argmin(dist, axis = 0)
pred_raw = np.argmin(dist_raw, axis = 0)
pred_encoded = np.argmin(dist_encoded, axis = 0)

for i in range(6):
    visualize_average(i*5)
    
plt.figure()
plt.imshow(x_test[37].reshape(img_height, img_width))
