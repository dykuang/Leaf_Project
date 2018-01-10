# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 23:25:23 2017

@author: dykua
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics

img_height, img_width = 64, 48
cls = 30
size = 340
batch_size = 128
epochs = 50
epsilon_std = 1

target_dir = 'data//leaf//'
leaf_data = np.load(target_dir+'leaf_data_{}_{}.npy'.format(img_height, img_width))
leaf_label = np.load(target_dir+'leaf_label_{}_{}.npy'.format(img_height, img_width))

leaf_data = leaf_data.reshape(size, img_height, img_width, 1)

input_img = Input(shape=(img_height, img_width,1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
z_mean = MaxPooling2D((2, 2), padding='same')(x)
z_log_var = MaxPooling2D((2, 2), padding='same')(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.,
                              stddev=epsilon_std)
#    epsilon = 0
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling)([z_mean, z_log_var])

De_Conv_1 = Conv2D(32, (3, 3), activation='relu', padding='same')
De_Upsam_1 = UpSampling2D((2, 2))
De_Conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')
De_Upsam_2 = UpSampling2D((2,2))
De_final = Conv2D(1, (3, 3), activation='sigmoid', padding='same')

x = De_Conv_1(z)
x = De_Upsam_1(x)
x = De_Conv_2(x)
x = De_Upsam_2(x)
decoded = De_final(x)


def vae_loss(input_img, decoded):
        xent_loss = img_height*img_width * metrics.binary_crossentropy(K.reshape(input_img, [-1]),
                                                                       K.reshape(decoded, [-1]))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    
vae = Model(input_img, decoded)
encoder = Model(input_img, z_mean)

# Add a decoder model
decoder_input = Input(shape = (16, 12, 32) )
y = De_Conv_1(decoder_input)
y = De_Upsam_1(y)
y = De_Conv_2(y)
y = De_Upsam_2(y)
y = De_final(y)
decoder = Model(decoder_input, y)


vae.compile(optimizer='rmsprop', loss='binary_crossentropy') # try a direct crossentropy loss? optimizer?

#from keras.callbacks import TensorBoard
vae.fit(leaf_data,leaf_data,
                batch_size = batch_size, 
                epochs = epochs,
                shuffle=True,
                verbose = 0)


                

data_test = leaf_data[20:120:10]
decoded_imgs = vae.predict(data_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(data_test[i].reshape(img_height, img_width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(img_height, img_width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#------------------------------------------------------------------------------
# Test cluster algorithms on encoder features
#------------------------------------------------------------------------------
x_test_encoded = encoder.predict(leaf_data, batch_size=batch_size)
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D

plt.figure(figsize=(10,10))
#features_std = preprocessing.scale(x_test_encoded)
features_std = x_test_encoded.reshape([len(x_test_encoded), np.prod(x_test_encoded.shape[1:])])
x_embedded_2d = Isomap(n_components=2).fit_transform(features_std)
plt.scatter(x_embedded_2d[:, 0], x_embedded_2d[:, 1], c=leaf_label)
plt.colorbar()

fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
x_embedded_3d = Isomap(n_components=3).fit_transform(features_std)
p = ax.scatter(x_embedded_3d[:, 0], x_embedded_3d[:, 1], x_embedded_3d[:,2], c=leaf_label)
fig.colorbar(p)
plt.show()
