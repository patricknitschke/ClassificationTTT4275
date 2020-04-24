import numpy as np


def sample_mean(dataset):
    N = len(dataset[0])
    x_sum = np.zeros((N,1))
    for vector in range (len(dataset)):
        for element in range (N):
            x_sum[element] += dataset[element][vector]
    for element in range(len(x_sum)):
        x_sum[element] /= N
    return x_sum

def cov_matrix(dataset): #CROSS CHECK THAT THIS IS CORRECT!!
    N = len(dataset[0])
    mean = sample_mean(dataset)
    mean = mean.reshape(1,N)
    cov_matrix = np.zeros((N,N))
    for sample in range (N):
        cov_matrix += (dataset[sample]-mean).T@(dataset[sample]-mean)
    return cov_matrix/N
'''
test = [[0.7,1],[2,3.4]]

print(cov_matrix(test))
'''



