# Leaf_Project
This repository contains some codes that I tried in classifying different leaves and possibly will be more organised once a good model is achieved. The data set for test can be obtained from [UCI's machine learning repository](https://archive.ics.uci.edu/ml/datasets/leaf), [Swedish leaf dataset](http://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/) and [UCI's 100 leaf](https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set). We intended to only use shape information currently.

# Current Best
The best model now is the architecture that looks like a "naive" module as google's Inception net but in 1D case. The input is just the CCD (center contour distance) feature, each conv1d block is with different kernel in order to extract possible features at different scale. These features are then concatenated to feed to following fully connected layers. Once this network is trained well, the classifying layer on top is taken off and replaced by a kernel svm trying to increase accuracy.

For all the three datasets, it can all get around or greater than 90% accurracy without tuning hyperparamters particularly. It can obtain >99% accuracy in the swedish leaf dataset (holding %10 as test). A pretrained model `leafconv1d.hdf5` is included.

# Brief description for each script
## Dependencies
`Keras`, `Tensorflow`, `scikit-learn`, `scikit-image`, `h5py`.

## utils
* `img2array.py`: Reading image files from each subfolder of the downloaded dataset, adjust the resolution according to requirement and save data into numpy array. Class labels are also produced accordingly.
* `transform_test.py`: Containing functions that help find proper affine transformations that tranforms the reference image to the target image with certain selectd cost functions. Currently, the order of the transformation is specified to *translation*, *rescale*(image center as the center) and *rotation*(image center as the center). 
* `img_align.py`: Aligning each image to the specified reference by transformations written in `transform_test.py`.

## feature extraction
* `leaf_CCD.py`: Reading numpy array produced by `img2array.py` and extract center contour distance(CCD) features.
* `Transer_VGG19_leaf.py`: Extracting/maxpooling output after the maxpooling layer of the second convolutional block from the pretrained VGG19 network as the feature vector. I tried to add training layers directly on top but the process stoped itself during training. Maybe it is because the hardware is not powerful to handle (on cpu of a laptop). So I decided to do it offine by first saving the extracted features and then later feed saved features to different classifiers for test.

## tests
* `VAE_leaf_dense.py`: Testing the performance of variational auto encoder using flattened fully connected layers on leaf data. Manifold learning is also used to visualize possible clusters in the latent space. Part of the code is from this [post](https://blog.keras.io/building-autoencoders-in-keras.html)
* `VAE_leaf_cnn.py`: Similar as above, but now with a CNN approach. 
* `encoder_clf_test.py`: This script compares the classification performance using nearest neighbor under three distances:
    * Pixel loss between test_img and an "class-average" image that is produced by directly taking the mean of training images with the same class label.
    * Same idea as above, but with the encoder feature trained from a VAE network as the "image".
    * Same idea as the first, but the "class-average" image is produced by decoding the mean of encoder features from each class. 
* `Keras_cnn_leaf.py`: Directly applying cnn on the leaf classification task. Since the data set is small, Keras's image generator is used.
* `feature_test.py`: Testing some common classifiers including: SVM, kNN, RandomForest and MLP on extraced features. Classifiers are from scikit-learn package.
* `leaf_pnn.py`: Testing SVM and PNN on extracted features.
* `leaf_1dconv.py`: The current best model using a google perception like module + svm.
