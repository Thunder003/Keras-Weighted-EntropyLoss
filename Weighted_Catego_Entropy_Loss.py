###FOR MSE CUSTOM CLASS WEIGHTS:- https://stackoverflow.com/questions/57840750/how-to-pass-weights-to-mean-squared-error-in-keras
#https://stackoverflow.com/questions/48082655/custom-weighted-loss-function-in-keras-for-weighing-each-element
#https://neptune.ai/blog/keras-loss-functions


from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import keras.backend as K
from itertools import product
from functools import partial

#THIS IS PUT EXTRA WEIGHT IF CLASS 1 IS MISCLASSIFIED AS 7 AND IF CLASS 7 IS MISCLASSIFIED AS 1


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    print('y_pred 1 is, ', y_pred)
    final_mask = K.zeros_like(y_pred[:, 0])
    print('final_mask 1 is, ', final_mask)
    y_pred_max = K.max(y_pred, axis=1)
    # print('y_pred_max 1 is, ', y_pred_max)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    
    # print('y_pred_max 2 is, ', y_pred_max)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    # print('y_pred_max_mat 1 is, ', y_pred_max_mat)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
          final_mask +=(K.cast(weights[c_t, c_p],tf.float32) * K.cast(y_pred_max_mat[:, c_p] ,tf.float32)* K.cast(y_true[:, c_t],tf.float32))
    print('Shape final_mask 1 is, ', final_mask.shape)
    print('final_mask 1 is, ', final_mask)
    print(' K.categorical_crossentropy(y_pred, y_true) ,', K.categorical_crossentropy(y_pred, y_true))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


w_array = np.ones((10,10))
w_array[1, 7] = 1.2
w_array[7, 1] = 1.2

ncce = partial(w_categorical_crossentropy, weights=w_array)

batch_size = 4
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss=ncce, optimizer=rms,run_eagerly=True)

model.fit(X_train, Y_train,
          batch_size=batch_size, epochs=nb_epoch, verbose=1,validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=1)