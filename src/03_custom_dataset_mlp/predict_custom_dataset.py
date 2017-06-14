"""Inference/predict code for MNIST

model must be trained before inference, train_mnist_4_trainer.py must be executed beforehand.
"""
from __future__ import print_function
import argparse
import time

import numpy as np
import six
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable, optimizers, serializers
from chainer import datasets, training, cuda, computational_graph
from chainer.dataset import concat_examples

from my_mlp import MyMLP
from my_dataset import MyDataset


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--modelpath', '-m', default='result/mlp.model',
                        help='Model path to be loaded')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=50,
                        help='Number of units')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of images in each mini-batch')
    args = parser.parse_args()

    batchsize = args.batchsize
    # Load the custom dataset
    dataset = MyDataset('data/my_data.csv')
    train_ratio = 0.7
    train_size = int(len(dataset) * train_ratio)
    train, test = chainer.datasets.split_dataset_random(dataset, train_size, seed=13)

    # Load trained model
    model = MyMLP(args.unit)  # type: MyMLP
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    xp = np if args.gpu < 0 else cuda.cupy

    serializers.load_npz(args.modelpath, model)

    x_list = []
    y_list = []
    t_list = []
    for i in range(0, len(test), batchsize):
        x, t = concat_examples(test[i:i + batchsize])
        y = model.predict(x)
        y_list.append(y.data)
        x_list.append(x)
        t_list.append(t)

    x_test = np.concatenate(x_list)[:, 0]
    y_test = np.concatenate(y_list)[:, 0]
    t_test = np.concatenate(t_list)[:, 0]
    print('x', x_test)
    print('y', y_test)
    print('t', t_test)

    plt.figure()
    plt.plot(x_test, t_test, 'o', label='test actual')
    plt.plot(x_test, y_test, 'o', label='test predict')
    plt.legend()
    plt.savefig('predict.png')


if __name__ == '__main__':
    main()
