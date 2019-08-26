# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:39:40 2017

@author: Job

Reference: How to add and remove new layers
https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights
Reference: How to freeze layers
https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
"""

import numpy as np
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras import optimizers
from keras.layers import Input
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda
from keras.callbacks import TensorBoard
from keras.utils import np_utils
import h5py
import datetime

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from stl import mesh
from stl_to_point_cloud import stl_to_point
from open_save_file import get_file_name, get_label

num_points = 2048
k = 3


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def mat_mul(A, B):
    return tf.matmul(A, B)


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def read_data(file_dir):
    all_points = None
    for file in file_dir:
        data = mesh.Mesh.from_file(file)
        point = stl_to_point(v1=data.v0, v2=data.v2, v3=data.v1, num_points=num_points)  # Order to get upright shape
        point = np.expand_dims(point, axis=0)
        if all_points is None:
            all_points = point
        else:
            all_points = np.concatenate((all_points, point), axis=0)
    return all_points


def transfer_train(X_train_name, X_test_name, y_train, y_test, model_name, model_weight_name):
    all_points = read_data(file_dir)

    X_test = read_data(X_test_name)
    # label to categorical
    y_train = (y_train - 1) / 2
    y_test = (y_test - 1) / 2
    y_train_categorial = np_utils.to_categorical(y_train, k)
    y_test_categorial = np_utils.to_categorical(y_test, k)

    """
    __________________________________________________________________________________________________
    conv1d_11 (Conv1D)              (None, 2048, 256)    33024       batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 2048, 256)    1024        conv1d_11[0][0]                  
    __________________________________________________________________________________________________
    max_pooling1d_3 (MaxPooling1D)  (None, 1, 256)       0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 1, 512)       131584      max_pooling1d_3[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 1, 512)       2048        dense_7[0][0]                    
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 1, 512)       0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    dense_8 (Dense)                 (None, 1, 256)       131328      dropout_1[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 1, 256)       1024        dense_8[0][0]                    
    __________________________________________________________________________________________________
    dropout_2 (Dropout)             (None, 1, 256)       0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    new_dense_9 (Dense)             (None, 1, 3)         771         dropout_2[0][0]                  
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 3)            0           new_dense_9[0][0]                
    ==================================================================================================
    """
    model = load_model(model_name, custom_objects={'mat_mul': mat_mul, })
    model.load_weights(model_weight_name)

    # Remove two last layer to fit new output type
    model.layers.pop()
    model.layers.pop()
    c = Dense(k, activation='softmax', name="new_dense_9")(model.layers[-1].output)
    prediction = Flatten()(c)
    # for i in range(train_layer):
    #     model.layers.pop()
    # c = Dense(256, activation='relu', name="new_dense_8")(model.layers[-1].output)
    # c = BatchNormalization(name="new_batch_normalization_17")(c)
    # c = Dropout(rate=0.7, name="new_dropout_2")(c)
    # c = Dense(k, activation='softmax', name="new_dense_9")(c)
    # prediction = Flatten()(c)

    input_points = Input(shape=(num_points, k))
    new_model = Model(inputs=model.input, outputs=[prediction])

    train_layer = 5
    for layer in new_model.layers[:-train_layer]:  # Freeze last few layers (Two dense layer) and is not batchnorm
        # if not isinstance(layer, K.layers.normalization.BatchNormalization):
        layer.trainable = False
    new_model.summary()
    for layer in new_model.layers:
        print(layer, layer.trainable)

    # define optimizer
    adam = optimizers.Adam(lr=0.001, decay=0.1)

    new_model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # compile classification new_model
    class_weight_val = compute_class_weight('balanced', np.unique(np.squeeze(y_train)), np.squeeze(y_train)).tolist()
    clamp_val = 4  # Max value cannot be more than 5 times of min
    if max(class_weight_val) / min(class_weight_val) > clamp_val:
        class_weight_val[class_weight_val.index(max(class_weight_val))] = clamp_val * min(
            class_weight_val)
    print(class_weight_val)
    class_weight = dict()
    for i, val in enumerate(class_weight_val):
        class_weight[i] = val

    epoch = 100

    # Adaptation Code, consume more memory
    X_train_original = read_data(X_train_name)
    X_train = X_train_original
    y_train_categorial_aug = y_train_categorial

    for i in range(1, epoch):
        X_train_new = jitter_point_cloud(X_train_original)
        X_train = np.concatenate((X_train, X_train_new), axis=0)
        y_train_categorial_aug = np.concatenate((y_train_categorial_aug, y_train_categorial), axis=0)

    training_history = new_model.fit(X_train, y_train_categorial_aug,
                                     validation_data=(X_test, y_test_categorial),
                                     batch_size=32, epochs=epoch,
                                     shuffle=True, verbose=1,
                                     callbacks=[TrainValTensorBoard(write_graph=False)],
                                     # callbacks=[tensorboard_callback],
                                     class_weight=class_weight, )
    print("Average test loss: ", np.average(training_history.history['loss']))

    score = new_model.evaluate(X_test, y_test_categorial, verbose=1)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

    '''
    # Original Code
    for i in range(0, epoch):
        X_train = read_data(X_train_name)
        # X_train = rotate_point_cloud(X_train)
        # X_train = jitter_point_cloud(X_train)
        training_history = new_model.fit(X_train, y_train_categorial, batch_size=32, epochs=1,
                                         shuffle=True, verbose=1, callbacks=[tensorboard_callback],
                                         class_weight=class_weight, )
        s = "Current epoch is:" + str(i)
        
        print(s)
        if i % 5 == 0:
            score = new_model.evaluate(X_test, y_test_categorial, verbose=1)
            print('Test loss: ', score[0])
            print('Test accuracy: ', score[1])
    '''

    prediction = new_model.predict(X_test, verbose=0)

    for i in range(np.shape(y_test)[0]):
        is_correct = np.argmax(prediction[i, :]) == y_test[i, :]
        print("Actual:%s, Prediction: %s, Confident: %s, Is it correct: %s" %
              (y_test[i, :], np.argmax(prediction[i, :]), prediction[i, np.argmax(prediction[i, :])], is_correct))

    prediction = new_model.predict(X_train, verbose=0)

    for i in range(np.shape(y_train)[0]):
        is_correct = np.argmax(prediction[i, :]) == y_train[i, :]
        print("Train Result: Actual:%s, Prediction: %s, Confident: %s, Is it correct: %s" %
              (y_train[i, :], np.argmax(prediction[i, :]), prediction[i, np.argmax(prediction[i, :])], is_correct))

    score = new_model.evaluate(X_test, y_test_categorial, verbose=1)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

    if score[1] > 0.6:
        new_model.save('transfered_model_.h5')
        new_model.save_weights('transfered_model_weights.h5')


if __name__ == '__main__':
    label_name_key = ["Occ_B", "Occ_F", "Occ_L", "Occ_Sum", "BL", "MD", "Taper_Sum", "Integrity", "Width", "Surface",
                      "Sharpness"]
    file_dir, _ = get_file_name('../global_data/stl_data', file_name='PreparationScan.stl')
    label, _ = get_label("MD", "median", double_data=False)
    label = np.asarray(label).reshape(-1, 1)
    all_points = None

    X_train_name, X_test_name, y_train, y_test = train_test_split(file_dir, label, test_size=0.2, random_state=0)

    model_name = 'base_model_50'
    transfer_train(X_train_name, X_test_name, y_train, y_test, model_name + '.h5', model_name + '_weights.h5')

    # TODO: Epoch: 50, 100, train all, train only batch, train only dense
