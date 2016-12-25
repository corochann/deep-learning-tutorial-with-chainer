from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import chainer

# Load the MNIST dataset from pre-inn chainer method
train, test = chainer.datasets.get_mnist()
#train, test = chainer.datasets.get_mnist(ndim=2)  ## ndim=2 from beginning

# train[i][0] represents x_i, MNIST image data,
# type=numpy(784,) vector <- specified by ndim of get_mnist()
print('train[0][0]', train[0][0].shape, train[0][0])
# train[i][0] represents y_i, MNIST label data(0-9), type=numpy() scalar
print('train[0][1]', train[0][1].shape, train[0][1])

for i in range(15):
    # train[i][0] is i-th image data with size 28x28
    image = train[i][0].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(3, 5, i+1)          # subplot with size (width 3, height 5)
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('label = {}'.format(train[i][1]))
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('mnist_plot.png')
