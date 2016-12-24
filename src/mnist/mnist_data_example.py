from __future__ import print_function

import numpy as np

import chainer

# Load the MNIST dataset from pre-inn chainer method
train, test = chainer.datasets.get_mnist()

# train[i] represents i-th data, there are 60000 training data
# test data structure is same, but total 10000 test data
print('len(train), type ', len(train), type(train))
print('len(test), type ', len(test), type(test))

# train[i] represents i-th data, type=tuple(x_i, y_i)
print('train[0]', type(train[0]), len(train[0]), train[0])

# train[i][0] represents x_i, MNIST image data,
# type=numpy(784,) vector <- specified by ndim of get_mnist()
print('train[0][0]', train[0][0].shape, train[0][0])
# train[i][0] represents y_i, MNIST label data(0-9), type=numpy() scalar
print('train[0][1]', train[0][1].shape, train[0][1])

# we cannot slice tupleDataset like this
print('train[0:3][0]', type(train[0:3][0]), train[0:3][0])
print('train[0:3][1]', type(train[0:3][1]), train[0:3][1])

# we can slice tupleDataset using numpy array
#nparray = np.asarray([0, 1, 2])

nparray = np.asarray(list(range(4, 8)))
print('train[[0,1,2]][0]', type(train[nparray][0]), train[nparray][0].shape, train[nparray][0])
print('train[[0,1,2]][1]', type(train[nparray][1]), train[nparray][1].shape, train[nparray][1])
print(list(range(4, 8)))


