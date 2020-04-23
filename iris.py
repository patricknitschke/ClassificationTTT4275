from sklearn.datasets import load_iris
import numpy as np
import random


iris = load_iris()
data = iris.data
target = iris.target
training_data1 = data[0:30]
training_data2 = data[51:81]
training_data3 = data[101:131]
training_data = [training_data1,training_data2,training_data3]
test = data
solution = target


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
    D = len(training_data[0][0])
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
        print(W)
    return W
    

def test_instance (W,x,solution,confusion):
  x = np.append(x,0)
  Wx = W@x
  answer = np.argmax(Wx)
  print("solution:",solution,"guess:",answer)
  confusion[solution][answer] += 1
  if solution == answer:
    return True
  return False



def test_sequence(W,x_sequence,solution_sequence,n_classes):
  confusion_matrix = np.zeros((n_classes,n_classes))
  n_tests = 60
  correct = 0
  wrong = 0
  
  for i in range (n_tests): 
    k = random.randint(0,130)
    if test_instance(W,x_sequence[k],solution_sequence[k],confusion_matrix):
      correct += 1
    else:
      wrong += 1
  return correct,wrong,correct/n_tests,confusion_matrix



W = find_W(training_data,5000,3,0.01)
print(W)


c,w,r,m = test_sequence(W,data,target,3)

print(r,m)




    


