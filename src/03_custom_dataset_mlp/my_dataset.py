import numpy as np
import pandas as pd

import chainer


class MyDataset(chainer.dataset.DatasetMixin):

    def __init__(self, filepath, debug=False):
        self.debug = debug
        # Load the data in initialization
        df = pd.read_csv(filepath)
        self.data = df.values.astype(np.float32)
        if self.debug:
            print('[DEBUG] data: \n{}'.format(self.data))

    def __len__(self):
        """return length of this dataset"""
        return len(self.data)

    def get_example(self, i):
        """Return i-th data"""
        x, t = self.data[i]
        return [x], [t]

if __name__ == '__main__':
    # Test code
    dataset = MyDataset('data/my_data.csv', debug=True)

    print('Access by index dataset[1] = ', dataset[1])
    print('Access by slice dataset[:3] = ', dataset[:3])
    print('Access by list dataset[[3, 5]] = ', dataset[[3, 5]])
    index = np.arange(3)
    print('Access by numpy array dataset[[0, 1, 2]] = ', dataset[index])
    # Randomly take 3 data
    index = np.random.permutation(len(dataset))[:3]
    print('dataset[{}] = {}'.format(index, dataset[index]))
