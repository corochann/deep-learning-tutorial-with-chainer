from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, iterators, serializers, optimizers
from chainer.training import extensions

from CNNSmall import CNNSmall
from CNNMedium import CNNMedium


def main():
    archs = {
        'cnnsmall': CNNSmall,
        'cnnmedium': CNNMedium,
    }

    parser = argparse.ArgumentParser(description='Cifar-100 CNN example')
    parser.add_argument('--arch', '-a', choices=archs.keys(),
                        default='cnnsmall', help='Convnet architecture')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result-cifar100',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # 1. Setup model
    class_num = 100
    model = archs[args.arch](n_out=class_num)
    classifier_model = L.Classifier(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        classifier_model.to_gpu()  # Copy the model to the GPU

    # 2. Setup an optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(classifier_model)

    # 3. Load the CIFAR-10 dataset
    train, test = chainer.datasets.get_cifar100()

    # 4. Setup an Iterator
    train_iter = iterators.SerialIterator(train, args.batchsize)
    test_iter = iterators.SerialIterator(test, args.batchsize,
                                         repeat=False, shuffle=False)

    # 5. Setup an Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    # 6. Setup a trainer (and extensions)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, classifier_model, device=args.gpu))

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

    trainer.extend(extensions.ProgressBar())

    # Resume from a snapshot
    if args.resume:
        serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()
    serializers.save_npz('{}/{}-cifar100.model'
                         .format(args.out, args.arch), model)

if __name__ == '__main__':
    main()


