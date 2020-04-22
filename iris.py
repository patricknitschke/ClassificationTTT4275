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


def sigmoid(g):
  return 1 / (1 + np.exp(-g))


def update_mse_grad(data,t,W,C):
  grad_mse = np.zeros((3,1))
  for i in range(len(data)):
    x = data[i]
    x = np.append(x,0)
    z = W@x
    g = sigmoid(z)
    calc = np.dot((g-t),g)
    calc = np.dot(calc,np.ones((1,C))-g)
    calc = calc.reshape((3,1))
    grad_mse += calc
  return grad_mse*x

def find_W(data, m_iterations,n_classes,alpha):
    C = n_classes
    D = len(training_data[0][0])
    W_x= np.ones((C,D))
    W_0 = np.zeros((C,1))
    W = np.concatenate((W_x,W_0),axis = 1)
    
    t_k1 = np.array([1,0,0]) #class1
    t_k2 = np.array([0,1,0]) #class2
    t_k3 = np.array([0,0,1]) #class3
    t  = [t_k1, t_k2, t_k3]

    for m in range(m_iterations):
        
        W_prev = W
        grad_mse = np.zeros((C,D+1))  
        update = grad_mse
        for t_k in t :
            for data_k in data:
              grad_mse += update_mse_grad(data_k,t_k,W_prev,C)
        print(grad_mse)
        W = W_prev - alpha*grad_mse
    return W
    

def test_instance (W,x,solution):
  x = np.append(x,0)
  Wx = np.dot(W,x)
  answer = np.argmax(sigmoid(Wx))
  #print("solution:",solution,"guess:",answer)
  if solution == answer:
    return True
  return False



def test_sequence(W,x_sequence,solution_sequence):
  n_tests = 20
  correct = 0
  wrong = 0
  
  for i in range (n_tests): 
    k = random.randint(0,150)
    if test_instance(W,x_sequence[k],solution_sequence[k]):
      correct += 1
    else:
      wrong += 1
  return correct,wrong,correct/n_tests



W = find_W(training_data,200,3,0.01)
print(W)


y,n,r = test_sequence(W,test,solution)

print (r)

    


