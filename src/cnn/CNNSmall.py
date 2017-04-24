import chainer
import chainer.functions as F
import chainer.links as L


class CNNSmall(chainer.Chain):
    def __init__(self, n_out):
        super(CNNSmall, self).__init__(
            conv1=L.Convolution2D(None, 16, 3, 2),
            conv2=L.Convolution2D(16, 32, 3, 2),
            conv3=L.Convolution2D(32, 32, 3, 2),
            fc4=L.Linear(None, 100),
            fc5=L.Linear(100, n_out)
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        h = self.fc5(h)
        return h
