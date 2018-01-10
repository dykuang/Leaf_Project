# -*- coding: utf-8 -*-
'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference
 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
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
latent_dim = 8    # if latent dimension is larger than 2, use a t-SNE to visualize
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
#    epsilon = .1
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h0 = Dense(64, activation='relu')
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='softmax')

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
vae.compile(optimizer='rmsprop', loss=None) # this optimizer seems to work best


# train the VAE on leaf data
leaf_data = leaf_data.reshape(size, img_height, img_width, 1)
#x_train, x_test, y_train, y_test = train_test_split(
#                             leaf_data, leaf_label-1, test_size=0.10, #careful here
#                             random_state=42,
#                             shuffle = True)

train_ind = np.arange(25,41)
x_train = leaf_data[train_ind,]
x_test = x_train
y_train = leaf_label[train_ind,]
y_test = leaf_label[train_ind,]

#==============================================================================
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     zoom_range=0.1,
#     rotation_range=40,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     vertical_flip=True)
# 
# 
# datagen.fit(x_train)
#==============================================================================

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#from keras.callbacks import TensorBoard
vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),
        verbose = 0)

#for e in range(epochs):
#    print('Epoch', e)
#    batches = 0
#    for x_batch in datagen.flow(x_train, None, batch_size=64):
#        x_batch = x_batch.reshape((64, np.prod(x_train.shape[1:])))
#        print(x_batch.shape)
#        vae.fit(x_batch, None)
#        batches += 1
#        if batches >= len(x_train) / 128:
#            # we need to break the loop by hand because
#            # the generator loops indefinitely
#            break

#vae.fit_generator(datagen.flow(x_train, None, batch_size=128),
#                    steps_per_epoch=len(x_train) / 128, epochs = 10,
#                    verbose=0, validation_data = (x_train,None)) 


# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test.reshape((len(x_test), original_dim)), batch_size=batch_size)
#==============================================================================
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 1], x_test_encoded[:, 2], c=y_test)
# plt.colorbar()
# plt.show()
#==============================================================================

# build a digit generator that can sample from the *learned* distribution
decoder_input = Input(shape=(latent_dim,))
_h0_decoded = decoder_h0(decoder_input)
_h_decoded = decoder_h(_h0_decoded)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

#==============================================================================
# # display a 2D manifold of the digits
# n = 15  # figure with 15x15 digits
# digit_size_x = img_height
# digit_size_y = img_width
# figure = np.zeros((digit_size_x * n, digit_size_y * n))
# # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# # to produce values of the latent variables z, since the prior of the latent space is Gaussian
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
# 
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi, 1, 1]])
#         x_decoded = generator.predict(z_sample)
#         digit = x_decoded[0].reshape(digit_size_x, digit_size_y)
#         figure[i * digit_size_x: (i + 1) * digit_size_x,
#                j * digit_size_y: (j + 1) * digit_size_y] = digit
# 
# plt.figure(figsize=(20, 20))
# plt.imshow(figure, cmap='Greys_r')
# plt.show()
#==============================================================================


#==============================================================================
# Visualize by t_SNE
#==============================================================================

from sklearn.manifold import TSNE, MDS, Isomap
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D

plt.figure(figsize=(10,10))
#features_std = preprocessing.scale(x_test_encoded)
features_std = x_test_encoded
x_embedded_2d = Isomap(n_components=2).fit_transform(features_std)
plt.scatter(x_embedded_2d[:, 0], x_embedded_2d[:, 1], c=y_test)
plt.colorbar()

fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
x_embedded_3d = Isomap(n_components=3).fit_transform(features_std)
p = ax.scatter(x_embedded_3d[:, 0], x_embedded_3d[:, 1], x_embedded_3d[:,2], c=y_test)
fig.colorbar(p)

fig = plt.figure()
check_ind = np.arange(25, 36)
g1 = encoder.predict(x_train)
g1_ave = np.mean(g1, axis = 0)
ave = generator.predict(g1_ave.reshape([1,latent_dim]))
#for i in range(ave.shape[1]):
#    if ave[0,i] < 0.6:
#        ave[0,i] = 0

ave_img = ave.reshape(img_height, img_width)
plt.subplot(121)
plt.imshow(ave_img)
plt.subplot(122)
plt.imshow(leaf_data[check_ind[0], :,:, 0])
plt.colorbar()
plt.show()