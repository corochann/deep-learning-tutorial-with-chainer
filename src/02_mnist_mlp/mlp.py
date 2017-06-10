import chainer
import chainer.functions as F
import chainer.links as L


# Network definition Chainer v2
# 1. `init_scope()` is used to initialize links for IDE friendly design.
# 2. input size of Linear layer can be omitted
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # input size of each layer will be inferred when omitted
            self.l1 = L.Linear(n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


# Network definition used in Chainer v1 (also available in Chainer v2)
class MLP1(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP1, self).__init__(
            # input size of each layer will be inferred when set `None`
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),    # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
