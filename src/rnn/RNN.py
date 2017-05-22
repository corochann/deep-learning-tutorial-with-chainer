import chainer
import chainer.functions as F
import chainer.links as L


class RNN(chainer.Chain):
    """Simple Recurrent Neural Network implementation"""
    def __init__(self, n_vocab, n_units):
        super(RNN, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.Linear(n_units, n_units),
            r1=L.Linear(n_units, n_units),
            l2=L.Linear(n_units, n_units),
            r2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.recurrent_h1 = None
        self.recurrent_h2 = None

    def reset_state(self):
        self.recurrent_h1 = None
        self.recurrent_h2 = None

    def __call__(self, x):
        h1 = self.embed(x)
        if self.recurrent_h1 is None:
            self.recurrent_h1 = F.tanh(self.l1(h1))
        else:
            self.recurrent_h1 = F.tanh(self.l1(h1) + self.r1(self.recurrent_h1))
        if self.recurrent_h2 is None:
            self.recurrent_h2 = F.tanh(self.l2(self.recurrent_h1))
        else:
            self.recurrent_h2 = F.tanh(self.l2(self.recurrent_h1) + self.r2(self.recurrent_h1))
        y = self.l3(self.recurrent_h2)
        return y
