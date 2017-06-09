import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer.dataset import concat_examples


class MLP2(chainer.Chain):

    def __init__(self, n_units):
        super(MLP2, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_units)  # n_units -> n_units
            self.l4 = L.Linear(1)    # n_units -> n_out

    def __call__(self, *args):
        self.forward(args)

    def forward(self, *args):
        x = args[0]
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        h = F.sigmoid(self.l3(h))
        h = self.l4(h)
        if chainer.config.train:  # Train phase
        #if True:
            t = args[1]
            # Calculate loss
            self.loss = F.mean_squared_error(h, t)
            reporter.report({'loss': self.loss}, self)
            return self.loss
        else:  # predict phase
            return h

    def predict(self, examples):
        minibatch = concat_examples(examples)
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                return self(*minibatch)
