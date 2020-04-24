import numpy as np
import extract_classes as ext


def sample_mean(dataset):
    sample_size = len(dataset)
    N = len(dataset[0])
    x_sum = np.zeros((N,1))
    for vector in range (sample_size):
        for element in range (N):
            if element != "":
                x_sum[element] += float(dataset[vector][element])
    for element in range(len(x_sum)):
        x_sum[element] /= sample_size
    return x_sum

def cov_matrix(dataset): #CROSS CHECK THAT THIS IS CORRECT!!
    N = len(dataset[0])
    mean = sample_mean(dataset)
   
    mean = mean.reshape(1,N)
    cov_matrix = np.zeros((N,N))
    for sample in range (N):
        x=np.asfarray(dataset[sample],float)
        cov_matrix += (x-mean).T@((x)-mean)
    return cov_matrix/N


def generate_mean_cov_map(filename):
    mean_cov_map = {}
    classes_map = ext.extract_classes_map(filename)
    for sound in classes_map:
        mean_cov_map[sound] =[sample_mean(classes_map[sound])]
        
        mean_cov_map[sound].append(cov_matrix(classes_map[sound]))
        

    return mean_cov_map

print(generate_mean_cov_map("data.dat"))



