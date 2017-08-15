"""Inference/predict code for simple_sequence dataset

model must be trained before inference, 
train_simple_sequence.py must be executed beforehand.
"""
from __future__ import print_function

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, iterators, serializers, optimizers, Variable, cuda
from chainer.training import extensions

sys.path.append(os.pardir)
from RNN import RNN
from RNN2 import RNN2
from RNN3 import RNN3
from RNNForLM import RNNForLM


def main():
    archs = {
        'rnn': RNN,
        'rnn2': RNN2,
        'rnn3': RNN3,
        'lstm': RNNForLM
    }

    parser = argparse.ArgumentParser(description='simple_sequence RNN predict code')
    parser.add_argument('--arch', '-a', choices=archs.keys(),
                        default='rnn', help='Net architecture')
    #parser.add_argument('--batchsize', '-b', type=int, default=64,
    #                    help='Number of images in each mini-batch')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--primeindex', '-p', type=int, default=1,
                        help='base index data, used for sequence generation')
    parser.add_argument('--length', '-l', type=int, default=100,
                        help='length of the generated sequence')
    parser.add_argument('--modelpath', '-m', default='',
                        help='Model path to be loaded')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    #print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    train, val, test = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)
    print('')

    # load vocabulary
    ptb_word_id_dict = chainer.datasets.get_ptb_words_vocabulary()
    ptb_id_word_dict = dict((v, k) for k, v in ptb_word_id_dict.items())

    # Model Setup
    model = archs[args.arch](n_vocab=n_vocab, n_units=args.unit)
    classifier_model = L.Classifier(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        classifier_model.to_gpu()  # Copy the model to the GPU
    xp = np if args.gpu < 0 else cuda.cupy

    if args.modelpath:
        serializers.load_npz(args.modelpath, model)
    else:
        serializers.load_npz('result/{}_ptb.model'.format(args.arch), model)

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

    predicted_text_list = [ptb_id_word_dict[i] for i in predicted_sequence]
    print('Predicted sequence: ', predicted_sequence)
    print('Predicted text: ', ' '.join(predicted_text_list))

if __name__ == '__main__':
    main()


