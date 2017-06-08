"""
RNN Training code with simple sequence dataset
"""
from __future__ import print_function

import os
import sys
import argparse

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
from simple_sequence.simple_sequence_dataset import N_VOCABULARY, get_simple_sequence
from parallel_sequential_iterator import ParallelSequentialIterator
from bptt_updater import BPTTUpdater



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
    print('')

    # 1. Setup model
    #model = archs[args.arch](n_vocab=N_VOCABRARY, n_units=args.unit)  # activation=F.leaky_relu
    model = archs[args.arch](n_vocab=N_VOCABULARY,
                             n_units=args.unit)  # , activation=F.tanh
    classifier_model = L.Classifier(model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        classifier_model.to_gpu()  # Copy the model to the GPU

    eval_classifier_model = classifier_model.copy()  # Model with shared params and distinct states
    eval_model = classifier_model.predictor

    # 2. Setup an optimizer
    optimizer = optimizers.Adam(alpha=0.0005)
    #optimizer = optimizers.MomentumSGD()
    optimizer.setup(classifier_model)

    # 3. Load dataset
    train = get_simple_sequence(N_VOCABULARY)
    test = get_simple_sequence(N_VOCABULARY)

    # 4. Setup an Iterator
    train_iter = ParallelSequentialIterator(train, args.batchsize)
    test_iter = ParallelSequentialIterator(test, args.batchsize, repeat=False)

    # 5. Setup an Updater
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    # 6. Setup a trainer (and extensions)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, eval_classifier_model,
                                        device=args.gpu,
                                        # Reset the RNN state at the beginning of each evaluation
                                        eval_hook=lambda _: eval_model.reset_state())
                   )

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
        x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
        x_key='epoch',
        file_name='accuracy.png'))

    # trainer.extend(extensions.ProgressBar())

    # Resume from a snapshot
    if args.resume:
        serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()
    serializers.save_npz('{}/{}_simple_sequence.model'
                         .format(args.out, args.arch), model)

if __name__ == '__main__':
    main()


