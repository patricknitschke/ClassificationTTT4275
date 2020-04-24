import numpy as np


def sample_mean(dataset):
    N = len(dataset[0])
    x_sum = np.zeros((N,1))
    for vector in range (len(dataset)):
        for element in range (N):
            x_sum[element] += dataset[element][vector]
    print(x_sum)
    for element in range(len(x_sum)):
        x_sum[element] /=N
    return x_sum


