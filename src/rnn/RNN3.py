import chainer
import chainer.functions as F
import chainer.links as L


class RecurrentConcatBlock(chainer.Chain):
    """Subblock for RNN
    
    It concatenates recurrent units instead of just adding sum.
    n_in cannot be None
    """
    def __init__(self, n_in, n_out, n_recurrent=None,
                 activation_out=F.tanh, activation_recurrent=F.tanh):
        assert n_in is not None
        if n_recurrent is None:
            n_recurrent = n_out
        self.n_recurrent = n_recurrent
        self.n_out = n_out
        self.activation_out = activation_out
        self.activation_recurrent = activation_recurrent
        super(RecurrentConcatBlock, self).__init__(
            l=L.Linear(n_in + n_recurrent, n_out + n_recurrent),
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
        h_out, h_recurrent = F.split_axis(y, [self.n_out,], axis=1)
        self.rh = self.activation_recurrent(h_recurrent)
        return self.activation_out(h_out)


class RNN3(chainer.Chain):
    """RNN implementation using RecurrentConcatBlock"""
    def __init__(self, n_vocab, n_units, n_recurrent=None, activation=F.tanh, train=True):
        super(RNN3, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            r1=RecurrentConcatBlock(n_units, n_units, n_recurrent=n_recurrent,
                                    activation_out=activation, activation_recurrent=activation),
            r2=RecurrentConcatBlock(n_units, n_units, n_recurrent=n_recurrent,
                                    activation_out=activation, activation_recurrent=activation),
            r3=RecurrentConcatBlock(n_units, n_units, n_recurrent=n_recurrent,
                                    activation_out=activation, activation_recurrent=activation),
            r4=RecurrentConcatBlock(n_units, n_units, n_recurrent=n_recurrent,
                                    activation_out=activation, activation_recurrent=activation),
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
