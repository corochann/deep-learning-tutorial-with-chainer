import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer.dataset import concat_examples


class MyMLP(chainer.Chain):

    def __init__(self, n_units):
        super(MyMLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_units)  # n_units -> n_units
            self.l4 = L.Linear(1)    # n_units -> n_out

    def __call__(self, *args):
        # Calculate loss
        h = self.forward(*args)
        t = args[1]
        self.loss = F.mean_squared_error(h, t)
        reporter.report({'loss': self.loss}, self)
        return self.loss

    def forward(self, *args):
        # Common code for both loss (__call__) and predict
        x = args[0]
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        h = F.sigmoid(self.l3(h))
        h = self.l4(h)
        return h

    def predict(self, *args):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                return self.forward(*args)

    def predict2(self, *args, batchsize=32):
        data = args[0]
        x_list = []
        y_list = []
        t_list = []
        for i in range(0, len(data), batchsize):
            x, t = concat_examples(data[i:i + batchsize])
            y = self.predict(x)
            y_list.append(y.data)
            x_list.append(x)
            t_list.append(t)

        x_array = np.concatenate(x_list)[:, 0]
        y_array = np.concatenate(y_list)[:, 0]
        t_array = np.concatenate(t_list)[:, 0]
        return x_array, y_array, t_array
