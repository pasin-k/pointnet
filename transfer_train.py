# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:39:40 2017

@author: Gary
"""

import numpy as np
import os
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda
from keras.utils import np_utils
import h5py

from sklearn.model_selection import train_test_split
from stl import mesh
from stl_to_point_cloud import stl_to_point
from open_save_file import get_file_name, get_label
num_point = 2048


def mat_mul(A, B):
    return tf.matmul(A, B)


if __name__ == '__main__':
    file_dir, _ = get_file_name('../global_data/stl_data', file_name='PreparationScan.stl')
    label, _ = get_label("BL","median",double_data=False)
    label = np.asarray(label).reshape(-1, 1)
    all_points = None

    for file in file_dir:
        data = mesh.Mesh.from_file(file)
        point = stl_to_point(v1=data.v0,v2=data.v1,v3=data.v2,num_points=num_point)
        point = np.expand_dims(point,axis=0)
        if all_points is None:
            all_points = point
        else:
            all_points = np.concatenate((all_points,point), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(all_points, label, test_size=0.2)
    print(np.shape(X_train))
    print(np.shape(X_test))
    print(np.shape(y_train))
    model = load_model('initial_model.h5')

    model.summary()
    model.layers.pop()
    model.layers.pop()
    model.summary()
    #TODO: Try remove layers + study pointnet more