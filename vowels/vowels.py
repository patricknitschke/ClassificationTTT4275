import numpy as np
import extract_classes as ext
from scipy.stats import multivariate_normal


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
    sample_size = len(dataset)
    mean = sample_mean(dataset)
   
    mean = mean.reshape(1,N)
    cov_matrix = np.zeros((N,N))
    for sample in range (N):
        x=np.asfarray(dataset[sample],float)
        cov_matrix += (x-mean).T@((x)-mean)/sample_size
    return cov_matrix


def generate_mean_cov_map(filename,start,end):
    mean_cov_map = {}
    classes_map = ext.extract_classes_map(filename)
    list_of_sounds = make_sequence(classes_map)
    for sound in classes_map:
        mean_cov_map[sound] =[sample_mean(classes_map[sound[start:end]])]
        
        mean_cov_map[sound].append(cov_matrix(classes_map[sound[start:end]]))
        

    return mean_cov_map,list_of_sounds

print(generate_mean_cov_map("data.dat",0,10))


def train_single_gaussian(start,end,x_size,diag = False):
    probabilities,list_sounds = np.zeros((12, x_size))
    mean_cov_map,list_sounds = generate_mean_cov_map("data.dat",start,end)
    for i,sound in enumerate(mean_cov_map):
        if diag == True:
            sound[2] = np.diag(sound[2])
        rv = multivariate_normal(sound[0],sound[2])
        probabilities[i] = rv

    return probabilities[i],list_sounds

def make_sequence(sounds):
    list_of_sounds= []
    for sound in sounds:
        list_of_sounds.append(sound)
    return list_of_sounds

def predict_not_diag(probability_sequence,sequence):
    probs,lelist = train_single_gaussian(0,30,15,False)
    i = np.argmax(probs)
    return lelist[i]
