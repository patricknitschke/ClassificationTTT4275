import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

classmap = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}

def arrange_order(x,t):
    '''
    Sets up the array such that the first 90 rows are 30*3 sets of different flowers
    '''
    array_x, array_t = [], []
    for i in range(50):
        for j in range(3):
            array_x.append(x[50*j+i])
            array_t.append(t[50*j+i])
    return np.array(array_x), np.array(array_t)

def get_x():
    f = open('./Iris_TTT4275/iris.data')
    x = np.loadtxt(f, delimiter=',', usecols=(0,1,2,3))
    f.close()
    # [x^T 1]^T -> x

    x_new = np.ones([np.size(x,0),np.size(x,1)+1])
    x_new[:,:-1] = x
    return x_new

def get_t():
    f = open('./iris_dataset/iris.data')
    t = np.loadtxt(f,delimiter=',', usecols=(4), dtype=str)
    f.close()
    t_updated = []
    for line in np.nditer(t):   
        t_updated.append(classmap.get(str(line))) # get target vectorial form
    return t_updated

def get_data(edit_order=True):
    '''
    Returns data from the dataset in mixed or correct order
    '''
    x = get_x()
    t = get_t()
    if edit_order:
        x,t = arrange_order(x,t)
    return x,t

def allocate_data(data, target):
	training_data1 = data[0:30]
	training_data2 = data[50:80]
	training_data3 = data[100:130]
	training_data = [training_data1,training_data2,training_data3]

	training_target1 = target[0:30]
	training_target2 = target[50:80]
	training_target3 = target[100:130]
	training_target = [training_target1, training_target2, training_target3]
	
	testing_data1 = data[30:50]
	testing_data2 = data[80:100]
	testing_data3 = data[130:150]
	testing_data = [testing_data1,testing_data2,testing_data3]

	testing_target1 = target[30:50]
	testing_target2 = target[80:100]
	testing_target3 = target[130:150]
	testing_target = [testing_target1, testing_target2, testing_target3]
	return training_data, training_target, testing_data, testing_target


# Approximation of the heaviside function
def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def update_mse_grad(test_data,t,W,C):
	grad_mse = np.zeros((C,len(test_data[0])+1))
	for i in range(len(test_data)):
		x = test_data[i]
		x = np.append(x,0)
		x = x.reshape(len(x),1)
		t = t.reshape(len(t),1)
		z = W@x
		g = sigmoid(z)
		grad_mse+= ((g-t)*(g*(1-g)))@x.T
	return grad_mse


def find_W(data, m_iterations,n_classes,alpha):
	C = n_classes
	D = len(data[0][0])
	W_x= np.zeros((C,D))
	W_0 = np.ones((C,1))
	W = np.concatenate((W_x,W_0),axis = 1)
	
	t_k1 = np.array([1,0,0]) #class1
	t_k2 = np.array([0,1,0]) #class2
	t_k3 = np.array([0,0,1]) #class3
	t  = [t_k1, t_k2, t_k3]

	for m in range(m_iterations):
		W_prev = W
		grad_mse = np.zeros((C,D+1))  
		for (data_k,t_k) in zip(data,t):
			grad_mse += update_mse_grad(data_k,t_k,W_prev,C)
		W = W_prev - alpha*grad_mse
	return W

def test_instance (W,x,solution,confusion):
	x = np.append(x,0)
	Wx = W@x
	answer = np.argmax(Wx)
	confusion[solution][answer] += 1
	if solution == answer:
		return True
	return False

def test_sequence(W,x_sequence,solution_sequence,n_classes,confusion_matrix):
	correct = 0
	wrong = 0
	for nclass in range (len(x_sequence)): 
		if test_instance(W,x_sequence[nclass],solution_sequence[nclass],confusion_matrix):
			correct += 1
		else:
			wrong += 1
	return correct,wrong,confusion_matrix


def assignment_1_trainingset(x_sequence,t_sequence,n_classes):
	confusion_matrix = np.zeros((3,3))
	W = find_W(x_sequence, 3000,n_classes, 0.01)
	tot = 0
	correct = 0
	wrong = 0
	for classes in range (3): 
		c,w,matrix = test_sequence(W,x_sequence[classes],t_sequence[classes],3,confusion_matrix)
		correct += c
		wrong += w
		tot += w+c
	return correct/tot,matrix,W


def assignment_1_testingset(W, training_data, testing_data, testing_solution, n_classes):
	confusion_matrix = np.zeros((3, 3))
	tot = 0
	correct = 0
	wrong = 0
	for classes in range(3):
		c,w,matrix = test_sequence(W, testing_data[classes], testing_solution[classes], n_classes, confusion_matrix)
		correct += c
		wrong += w
		tot += w+c
	return correct/tot, matrix



def printHistograms(data, features):
	for i in range(0,4):
		plt.figure(i)
		for j in range(3):
			correctdata = data[50*j : 50*(j+1), i]
			sns.distplot(correctdata, kde=True, norm_hist=True)
			labels = 'setosa', 'veriscolor', 'virginica'
			title = 'histogram for feature: ' + features[i]
			plt.legend(labels)
			plt.title(title)
	plt.show()

if __name__ == '__main__':
    print(get_x())