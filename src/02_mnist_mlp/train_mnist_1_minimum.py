"""
Very simple implementation for MNIST training code with Chainer using
Multi Layer Perceptron (MLP) model

This code is to explain the basic of training procedure.

"""
from __future__ import print_function
import time
import os
import numpy as np
import six

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import serializers


class MLP(chainer.Chain):
    """Neural Network definition, Multi Layer Perceptron"""
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred when `None`
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y


class SoftmaxClassifier(chainer.Chain):
    """Classifier is for calculating loss, from predictor's output.
    predictor is a model that predicts the probability of each label.
    """
    def __init__(self, predictor):
        super(SoftmaxClassifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss


def main():
    # Configuration setting
    gpu = -1                  # GPU ID to be used for calculation. -1 indicates to use only CPU.
    batchsize = 100           # Minibatch size for training
    epoch = 20                # Number of training epoch
    out = 'result/1'  # Directory to save the results
    unit = 50                 # Number of hidden layer units, try incresing this value and see if how accuracy changes.

    print('GPU: {}'.format(gpu))
    print('# unit: {}'.format(unit))
    print('# Minibatch-size: {}'.format(batchsize))
    print('# epoch: {}'.format(epoch))
    print('out directory: {}'.format(out))

    # Set up a neural network to train
    model = MLP(unit, 10)
    # Classifier will calculate classification loss, based on the output of model
    classifier_model = SoftmaxClassifier(model)

    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()  # Make a specified GPU current
        classifier_model.to_gpu()           # Copy the model to the GPU
    xp = np if gpu < 0 else cuda.cupy

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(classifier_model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    n_epoch = epoch
    N = len(train)       # training data size
    N_test = len(test)  # test data size

    # Learning loop
    for epoch in range(1, n_epoch + 1):
        print('epoch', epoch)

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        start = time.time()
        for i in six.moves.range(0, N, batchsize):
            x = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][0]))
            t = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][1]))

            # Pass the loss function (Classifier defines it) and its arguments
            optimizer.update(classifier_model, x, t)

            sum_loss += float(classifier_model.loss.data) * len(t.data)
            sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)
        end = time.time()
        elapsed_time = end - start
        throughput = N / elapsed_time
        print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
            sum_loss / N, sum_accuracy / N, throughput))

        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N_test, batchsize):
            index = np.asarray(list(range(i, i + batchsize)))
            x = chainer.Variable(xp.asarray(test[index][0]))
            t = chainer.Variable(xp.asarray(test[index][1]))

            loss = classifier_model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))

    # Save the model and the optimizer
    if not os.path.exists(out):
        os.makedirs(out)
    print('save the model')
    serializers.save_npz('{}/classifier_mlp.model'.format(out), classifier_model)
    serializers.save_npz('{}/mlp.model'.format(out), model)
    print('save the optimizer')
    serializers.save_npz('{}/mlp.state'.format(out), optimizer)

if __name__ == '__main__':
    main()

