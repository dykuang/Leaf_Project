# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 19:31:31 2017

@author: dykua

This script test different features that can be used

in leaf shape classification

First: central contour distance -----  MLP and SVM has ~60% test accuracy
        standardlize the input feature is important!
        SVM benefits with pca_components  = 5, linear kernel and balanced weight
        MLP uses 2 hidden layers each with 50 units, with 'relu' activation


Second: feature extracted from vgg19 after maxpooling:
       Not very good, both at around 47%

"""
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score

img_height, img_width = 480, 360
cls = 30
size = 340

target_dir = 'data//leaf//'
leaf_data = np.load(target_dir+'leaf_data_{}_{}_CCD.npy'.format(img_height, img_width))
#leaf_data = np.load(target_dir+'leaf_data_vgg19.npy')
leaf_label = np.load(target_dir+'leaf_label_{}_{}.npy'.format(img_height, img_width))


#leaf_data = ((leaf_data.transpose() - np.mean(leaf_data, axis = 1))/np.max(leaf_data, axis = 1))

x_train, x_test, y_train, y_test = train_test_split(
                             leaf_data, leaf_label-1, test_size=0.10, #careful here
                             random_state=42,
                             shuffle = True)
#------------------------------------------------------------------------------
#Preprocess: normalization, pca, kernel pca?
#------------------------------------------------------------------------------
scaler = RobustScaler().fit(x_train)
x_train_std = scaler.transform(x_train) 
x_test_std = scaler.transform(x_test)

#from sklearn.decomposition import PCA
#pre_pca = PCA(n_components=5).fit(x_train_std) # make it possible 
#x_train_pca = pre_pca.transform(x_train_std)
#x_test_pca = pre_pca.transform(x_test_std)
#
#from sklearn.preprocessing import MinMaxScaler
#mmscaler = MinMaxScaler((-1,1)).fit(x_train)
#x_train_mm = mmscaler.transform(x_train)
#x_test_mm = mmscaler.transform(x_test)



#------------------------------------------------------------------------------
#Tune the parameter for better performance. SVC
#------------------------------------------------------------------------------
from sklearn import svm  
clf = svm.SVC(C=1.1,cache_size=200, class_weight='balanced', coef0=0,
    decision_function_shape='ovr', degree=2, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

clf.fit(x_train_std, y_train)

y_pred = clf.predict(x_test_std)
#roc = roc_curve(y_test, y_pred)

print('accruracy of SVM: %.4f' % accuracy_score(y_test, y_pred) )


#------------------------------------------------------------------------------
# Random-Forest, Adaboost
#------------------------------------------------------------------------------
#==============================================================================
# from sklearn.ensemble import RandomForestClassifier
# clf_rf = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',
#             max_depth=2, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#             oob_score=False, random_state=0, verbose=0, warm_start=False)
# 
# clf_rf.fit(x_train, y_train)
# 
# y_pred_rf = clf_rf.predict(x_test)
# 
# print('accruracy of RF: %.4f' % accuracy_score(y_test, y_pred_rf) )
#==============================================================================

#------------------------------------------------------------------------------
# K-nn
#------------------------------------------------------------------------------
#==============================================================================
# from sklearn.neighbors import KNeighborsClassifier
# clf_knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto',
#                                leaf_size=30, p=2, metric='minkowski', 
#                                metric_params=None, n_jobs=1)
# 
# clf_knn.fit(x_train, y_train)
# 
# y_pred_knn = clf_knn.predict(x_test)
# 
# print('accruracy of knn: %.4f' % accuracy_score(y_test, y_pred_knn) )
#==============================================================================

#------------------------------------------------------------------------------
# PNN
#------------------------------------------------------------------------------
from sklearn.neural_network import MLPClassifier
clf_mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(50, 50), learning_rate='adaptive',
       learning_rate_init=0.005, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

clf_mlp.fit(x_train_std, y_train)

y_pred_mlp = clf_mlp.predict(x_test_std)

print('accruracy of MLP: %.4f' % accuracy_score(y_test, y_pred_mlp) )


