import numpy as np
import extract_classes as ext
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as GMM
import vowels_task1 as v


def map_join_array(type_map):
    x = []
    for sound in type_map:
        x.extend(type_map[sound[0:70]])
    return np.asfarray(x)
def generate_sound_list(train_map):
    sounds = []
    for sound in train_map:
        sounds.append(sound)
    return sounds

def train_test_GMM(start,end, n_components):
    train_map,test_map = v.generate_x("data.dat",0,70)
    sound_list = generate_sound_list(train_map)
    #---------------------Training------------------------------
    
    correct =  0
    wrong = 0
    total = 0
    confusion_matrix_train = np.zeros((12,12))
    x = map_join_array(train_map)
    print("Training GMM")
    probability_vectors = np.empty((12,x.shape[0]))
    

    for i,sound in enumerate(train_map):
        gmm = GMM(n_components=n_components, covariance_type='diag', reg_covar=1e-4, random_state=0)
        gmm.fit(train_map[sound], sound_list)
        for j in range(n_components):
            N = multivariate_normal(mean=gmm.means_[j], cov=gmm.covariances_[j], allow_singular=True)
            probability_vectors[i] += gmm.weights_[j] * N.pdf(x)
    
    predicted_indeces = np.argmax(probability_vectors,axis = 0)
    true = np.asarray([i for i in range (12) for _ in range(70)])
    for index in range(len(predicted_indeces)):
        if int(predicted_indeces[index]) == true[index]:
                correct += 1
        else:
            wrong += 1
        confusion_matrix_train[true[index]][int(predicted_indeces[index])] += 1
        total += 1


    print("Training : ")
    print(confusion_matrix_train)
    print(correct/total)
    return confusion_matrix_train



