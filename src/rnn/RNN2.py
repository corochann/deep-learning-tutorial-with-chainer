import chainer
import chainer.functions as F
import chainer.links as L


class RecurrentBlock(chainer.Chain):
    """Subblock for RNN"""
    def __init__(self, n_in, n_out, activation=F.tanh):
        super(RecurrentBlock, self).__init__(
            l=L.Linear(n_in, n_out),
            r=L.Linear(n_in, n_out),
        )
        self.rh = None
        self.activation = activation

    def reset_state(self):
        self.rh = None

    def __call__(self, h):
        if self.rh is None:
            self.rh = self.activation(self.l(h))
        else:
            self.rh = self.activation(self.l(h) + self.r(self.rh))
        return self.rh


class RNN2(chainer.Chain):
    """RNN implementation using RecurrentBlock"""
    def __init__(self, n_vocab, n_units, activation=F.tanh, train=True):
        super(RNN2, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            r1=RecurrentBlock(n_units, n_units, activation=activation),
            r2=RecurrentBlock(n_units, n_units, activation=activation),
            r3=RecurrentBlock(n_units, n_units, activation=activation),
            r4=RecurrentBlock(n_units, n_units, activation=activation),
            l5=L.Linear(n_units, n_vocab),
        )

    def reset_state(self):
        self.r1.reset_state()
        self.r2.reset_state()
        self.r3.reset_state()
        self.r4.reset_state()

    def __call__(self, x):
        h = self.embed(x)
        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)
        h = self.r4(h)
        y = self.l5(h)
        return y
