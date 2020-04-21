from sklearn.datasets import load_iris
import numpy as np
import math

iris = load_iris()
data = iris.data
target = iris.target
training_data1 = data[0:10]
training_data2 = data[51:61]
training_data3 = data[101:111]
training_data = [training_data1,training_data2,training_data3]
n_iterations = 100

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def find_W(data, m_iterations,n_classes,alpha):
    N = len(data)
    C = n_classes
    D = len(training_data[0][0])
    W_x = np.zeros((C,D))
    W_0 = np.zeros((C,1))
    W = np.concatenate((W_x,W_0),axis = 1)
    
    t_k1 = np.array([[1],[0],[0]]) #class1
    t_k2 = np.array([[0],[1],[0]]) #class2
    t_k3 = np.array([[0],[0],[1]]) #class3
    t_k  = [t_k1, t_k2, t_k3]

    for m in range(m_iterations):
        W_prev = W
        grad_mse = np.zeros((C,D+1))
        for k in range (0,N):
            
            for (data,t) in zip(data,t_k):
                x_k = data[k]
                x_k = np.append(x_k,1)
                x_k = x_k.reshape(D+1,1)
                print(x_k)
                z_k = W@x_k
                g_k = sigmoid(z_k)
                grad_mse_calculation = np.multiply(g_k-t,g_k)
                grad_mse_calculation = np.multiply(grad_mse_calculation,np.ones((1,C))-g_k)
                grad_mse +=grad_mse_calculation@x_k.T



find_W(training_data,10,3,0.5)

    


