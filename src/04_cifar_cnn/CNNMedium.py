import chainer
import chainer.functions as F
import chainer.links as L


class CNNMedium(chainer.Chain):
    def __init__(self, n_out):
        super(CNNMedium, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3, 1)
            self.conv2 = L.Convolution2D(16, 32, 3, 2)
            self.conv3 = L.Convolution2D(32, 32, 3, 1)
            self.conv4 = L.Convolution2D(32, 64, 3, 2)
            self.conv5 = L.Convolution2D(64, 64, 3, 1)
            self.conv6 = L.Convolution2D(64, 128, 3, 2)
            self.fc7 = L.Linear(None, 100)
            self.fc8 = L.Linear(100, n_out)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return h
