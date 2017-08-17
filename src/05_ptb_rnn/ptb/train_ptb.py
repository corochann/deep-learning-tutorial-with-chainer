"""
RNN Training code with Penn Treebank (ptb) dataset

Ref: https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb.py
"""
from __future__ import print_function

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, iterators, serializers, optimizers
from chainer.training import extensions

sys.path.append(os.pardir)
from RNN import RNN
from RNN2 import RNN2
from RNN3 import RNN3
from RNNForLM import RNNForLM
from parallel_sequential_iterator import ParallelSequentialIterator
from bptt_updater import BPTTUpdater


# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def main():
    archs = {
        'rnn': RNN,
        'rnn2': RNN2,
        'rnn3': RNN3,
        'lstm': RNNForLM
    }

    parser = argparse.ArgumentParser(description='RNN example')
    parser.add_argument('--arch', '-a', choices=archs.keys(),
                        default='rnn', help='Net architecture')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of RNN units in each layer')
    parser.add_argument('--bproplen', '-l', type=int, default=20,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Architecture: {}'.format(args.arch))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))

    # 1. Load dataset: Penn Tree Bank long word sequence dataset
    train, val, test = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1  # train is just an array of integers
    print('# vocab: {}'.format(n_vocab))
    print('')

    # 2. Setup model
    model = archs[args.arch](n_vocab=n_vocab,
                             n_units=args.unit)  # , activation=F.tanh
    classifier_model = L.Classifier(model)
    classifier_model.compute_accuracy = False  # we only want the perplexity

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        classifier_model.to_gpu()  # Copy the model to the GPU

    eval_classifier_model = classifier_model.copy()  # Model with shared params and distinct states
    eval_model = classifier_model.predictor

    # 2. Setup an optimizer
    optimizer = optimizers.Adam(alpha=0.001)
    #optimizer = optimizers.MomentumSGD()
    optimizer.setup(classifier_model)

    # 4. Setup an Iterator
    train_iter =ParallelSequentialIterator(train, args.batchsize)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)

    # 5. Setup an Updater
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    # 6. Setup a trainer (and extensions)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(val_iter, eval_classifier_model,
                                        device=args.gpu,
                                        # Reset the RNN state at the beginning of each evaluation
                                        eval_hook=lambda _: eval_model.reset_state())
                   )

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
    interval = 500
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity', 'elapsed_time']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.PlotReport(
        ['perplexity', 'val_perplexity'],
        x_key='epoch', file_name='perplexity.png'))

    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Resume from a snapshot
    if args.resume:
        serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()
    serializers.save_npz('{}/{}_ptb.model'
                         .format(args.out, args.arch), model)

    # Evaluate the final model
    print('test')
    eval_model.reset_state()
    evaluator = extensions.Evaluator(test_iter, eval_classifier_model, device=args.gpu)
    result = evaluator()
    print('test perplexity:', np.exp(float(result['main/loss'])))


if __name__ == '__main__':
    main()


