

N_VOCABRARY = 5

def get_simple_sequence(n_vocab, repeat=100):
    data = []
    for i in range(repeat):
        for j in range(n_vocab):
            for k in range(j):
                data.append(j)
    print(data)
    return data

get_simple_sequence(N_VOCABRARY)
