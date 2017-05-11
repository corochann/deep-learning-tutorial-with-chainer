import numpy as np

N_VOCABRARY = 10

def get_simple_sequence(n_vocab, repeat=100):
    data = []
    for i in range(repeat):
        for j in range(n_vocab):
            for k in range(j):
                data.append(j)

    return np.asarray(data, dtype=np.int32)

if __name__ == '__main__':
    data = get_simple_sequence(N_VOCABRARY)
    print(data)
