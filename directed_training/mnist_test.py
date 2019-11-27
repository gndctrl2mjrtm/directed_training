'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import pickle
import ray
import cv2

batch_size = 128
num_classes = 10
epochs = 12

def get_mnist():
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

@ray.remote
def calculate_score(z, n=100):
    x_train, y_train, _, __ = get_mnist()
    x_train = x_train[:n]
    y_train = y_train[:n]
    score = 0
    m = x_train[z]
    for i in range(n):
        if z != i:
            temp = x_train[i]
            score += np.linalg.norm(m-temp)
    del x_train, y_train
    return score/n


def order_dataset(n=100):
    x_train, y_train, _, __ = get_mnist()
    if n > len(x_train):
        n = len(x_train)
    x_train = x_train[:n]
    y_train = y_train[:n]

    scores = ray.get([calculate_score.remote(i, n=n) for i in range(n)])

    data = [[x,y] for x,y in zip(x_train,y_train)]
    sorted_data = [x for _,x in sorted(zip(scores,data))]

    sorted_x = []
    sorted_y = []

    for (x,y) in sorted_data:
        sorted_x.append(x)
        sorted_y.append(y)
    return sorted_x, sorted_y


def main():
    ray.init()
    sorted_x, sorted_y = order_dataset(n=60000)

    with open('sorted_x.pkl', 'wb') as f:
        pickle.dump(sorted_x, f)

    with open('sorted_y.pkl', 'wb') as f:
        pickle.dump(sorted_y, f)

if __name__ == "__main__":
    main()

