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


class RNNBackup(chainer.Chain):
    """Simple Recurrent Neural Network implementation"""
    def __init__(self, n_vocab, n_units):
        super(RNNBackup, self).__init__(
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
            xp = self.xp
            self.rh = xp.zeros_like(h.data)
        self.rh = self.activation(self.l(h) + self.r(self.rh))
        return self.rh


class RecurrentBlock2(chainer.Chain):
    """Subblock for RNN for backup"""
    def __init__(self, n_in, n_out):
        super(RecurrentBlock2, self).__init__(
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


class RecurrentConcatBlock(chainer.Chain):
    """Subblock for RNN
    
    It concatenates recurrent units instead of just adding sum.
    n_in cannot be None
    """
    def __init__(self, n_in, n_out, n_recurrent=None):
        assert n_in is not None
        if n_recurrent is None:
            n_recurrent = n_out
        self.n_recurrent = n_recurrent
        super(RecurrentConcatBlock, self).__init__(
            l=L.Linear(n_in + n_recurrent, n_out),
            r=L.Linear(n_in + n_recurrent, n_recurrent),
        )
        self.rh = None

    def reset_state(self):
        self.rh = None

    def __call__(self, h):
        if self.rh is None:
            xp = self.xp
            self.rh = xp.zeros((h.shape[0], self.n_recurrent), dtype=xp.float32)
        h_concat = F.concat((h, self.rh), axis=1)
        y = self.l(h_concat)
        self.rh = self.r(h_concat)
        return y


class RNN3(chainer.Chain):
    """RNN implementation using RecurrentConcatBlock"""
    def __init__(self, n_vocab, n_units, n_recurrent=None, train=True):
        super(RNN3, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            r1=RecurrentConcatBlock(n_units, n_units, n_recurrent=n_recurrent),
            r2=RecurrentConcatBlock(n_units, n_units, n_recurrent=n_recurrent),
            r3=RecurrentConcatBlock(n_units, n_units, n_recurrent=n_recurrent),
            r4=RecurrentConcatBlock(n_units, n_units, n_recurrent=n_recurrent),
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
