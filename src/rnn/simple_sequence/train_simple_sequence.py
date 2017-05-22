from __future__ import print_function
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, iterators, serializers, optimizers
from chainer.training import extensions

#from src.rnn.RNN import RNN, RNN2
from RNN import RNN, RNN2, RNN3
from simple_sequence_dataset import N_VOCABRARY, get_simple_sequence

# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


def main():
    archs = {
        'rnn': RNN,
        'rnn2': RNN2,
        'rnn3': RNN3,
    }

    parser = argparse.ArgumentParser(description='RNN example')
    parser.add_argument('--arch', '-a', choices=archs.keys(),
                        default='rnn2', help='Net architecture')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--bproplen', '-l', type=int, default=5,
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
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # 1. Setup model
    #model = archs[args.arch](n_vocab=N_VOCABRARY, n_units=args.unit)  # activation=F.leaky_relu
    model = archs[args.arch](n_vocab=N_VOCABRARY,
                             n_units=args.unit, activation=F.tanh)
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
    train = get_simple_sequence(N_VOCABRARY)
    test = get_simple_sequence(N_VOCABRARY)

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


