import chainer
import chainer.functions as F
import chainer.links as L


class RecurrentBlock(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(RecurrentBlock, self).__init__(
            l=L.Linear(n_in, n_out),
            r=L.Linear(n_in, n_out),
        )
        self.rh = 0

    def reset_state(self):
        self.rh = 0

    def __call__(self, h):
        y = self.l(h + self.rh)
        self.rh = self.r(h)
        return y


class RNN2(chainer.Chain):
    def __init__(self, n_vocab, n_units, train=True):
        super(RNN2, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            r1=RecurrentBlock(n_units, n_units),
            r2=RecurrentBlock(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )

    def reset_state(self):
        self.r1.reset_state()
        self.r2.reset_state()

    def __call__(self, x):
        h = self.embed(x)
        h = self.r1(h)
        h = self.r2(h)
        y = self.l3(h)
        return y


class RNN(chainer.Chain):
    def __init__(self, n_vocab, n_units):
        super(RNN, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.Linear(n_units, n_units),
            r1=L.Linear(n_units, n_units),
            l2=L.Linear(n_units, n_units),
            r2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.recurrent_h0 = 0
        self.recurrent_h1 = 0

    def reset_state(self):
        self.recurrent_h0 = 0
        self.recurrent_h1 = 0

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(h0 + self.recurrent_h0)
        self.recurrent_h0 = self.r1(h0)
        h2 = self.l2(h1 + self.recurrent_h1)
        self.recurrent_h1 = self.r2(h1)
        y = self.l3(h2)
        return y







