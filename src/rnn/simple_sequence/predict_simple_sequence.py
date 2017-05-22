"""Inference/predict code for simple_sequence dataset

model must be trained before inference, 
train_simple_sequence.py must be executed beforehand.
"""
from __future__ import print_function
import os
import argparse

import numpy as np
import matplotlib

from simple_sequence_dataset import N_VOCABRARY, get_simple_sequence

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, iterators, serializers, optimizers, Variable, cuda
from chainer.training import extensions

from RNN import RNN, RNN2, RNN3


def main():
    archs = {
        'rnn': RNN,
        'rnn2': RNN2,
        'rnn3': RNN3,
    }

    parser = argparse.ArgumentParser(description='simple_sequence RNN predict code')
    parser.add_argument('--arch', '-a', choices=archs.keys(),
                        default='rnn2', help='Net architecture')
    #parser.add_argument('--batchsize', '-b', type=int, default=64,
    #                    help='Number of images in each mini-batch')
    parser.add_argument('--primeindex', '-p', type=int, default=1,
                        help='base index data, used for sequence generation')
    parser.add_argument('--length', '-l', type=int, default=100,
                        help='length of the generated sequence')
    parser.add_argument('--modelpath', '-m', default='result/rnn2_simple_sequence.model',
                        help='Model path to be loaded')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    #print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    # Model Setup
    model = archs[args.arch](n_vocab=N_VOCABRARY, n_units=args.unit, activation=F.tanh)
    classifier_model = L.Classifier(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        classifier_model.to_gpu()  # Copy the model to the GPU
    xp = np if args.gpu < 0 else cuda.cupy

    serializers.load_npz(args.modelpath, model)

    # Dataset preparation
    prev_index = args.primeindex

    # Predict
    predicted_sequence = [prev_index]
    for i in range(args.length):
        prev = chainer.Variable(xp.array([prev_index], dtype=xp.int32))
        current = model(prev)
        current_index = np.argmax(cuda.to_cpu(current.data))
        predicted_sequence.append(current_index)
        prev_index = current_index

    print('Predicted sequence: ', predicted_sequence)

if __name__ == '__main__':
    main()


