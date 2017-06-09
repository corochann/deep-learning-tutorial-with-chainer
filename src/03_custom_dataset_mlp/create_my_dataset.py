import os
import numpy as np
import pandas as pd


DATA_DIR = 'data'


def black_box_fn(x_data):
    return np.sin(x_data)


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    x = np.arange(-1, 1, 0.01)
    t = black_box_fn(x)
    df = pd.DataFrame({'x': x, 't': t}, columns={'x', 't'})
    df.to_csv(os.path.join(DATA_DIR, 'my_data.csv'), index=False)
