import chainer
import chainer.functions as F
import chainer.links as L


class RNN(chainer.Chain):
    """Simple Recurrent Neural Network implementation"""
    def __init__(self, n_vocab, n_units):
        super(RNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.Linear(n_units, n_units)
            self.r1 = L.Linear(n_units, n_units)
            self.l2 = L.Linear(n_units, n_vocab)
        self.recurrent_h = None

    def reset_state(self):
        self.recurrent_h = None

    def __call__(self, x):
        h = self.embed(x)
        if self.recurrent_h is None:
            self.recurrent_h = F.tanh(self.l1(h))
        else:
            self.recurrent_h = F.tanh(self.l1(h) + self.r1(self.recurrent_h))
        y = self.l2(self.recurrent_h)
        return y
