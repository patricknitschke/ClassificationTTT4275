#Task 1 a and b

from sklearn.datasets import load_iris
import numpy as np
import random

import iris

iris_data = load_iris()
target = iris_data.target
data = iris_data.data

def task_a():
    training_data1 = data[0:30]
    training_data2 = data[50:80]
    training_data3 = data[100:130]
    training_data = [training_data1,training_data2,training_data3]
    training_target1 = target[0:30]
    training_target2 = target[50:80]
    training_target3 = target[100:130]
    training_target = [training_target1, training_target2, training_target3]

    test = data
    solution = target
    confusion_matrix = np.zeros((3,3))

    testing_data1 = data[30:50]
    testing_data2 = data[80:100]
    testing_data3 = data[130:150]
    testing_data = [testing_data1,testing_data2,testing_data3]
    testing_target1 = target[30:50]
    testing_target2 = target[80:100]
    testing_target3 = target[130:150]
    testing_target = [testing_target1, testing_target2, testing_target3]

    #------------------Training-------------------
    training_ratio, confusion_training, W = iris.assignment_1_trainingset(training_data,training_target,3,confusion_matrix)
    print("Training:")
    print(training_ratio)
    print(confusion_training)

    #-------------------Testing---------------------
    test_ratio, confusion_test = iris.assignment_1_testingset(W, training_data, testing_data, testing_target, 3, confusion_matrix)
    print("Testing:")
    print(test_ratio)
    print(confusion_test)  


def task_b():
    training_data1 = data[20:50]
    training_data2 = data[70:100]
    training_data3 = data[120:150]
    training_data = [training_data1,training_data2,training_data3]
    training_target1 = target[20:50]
    training_target2 = target[70:100]
    training_target3 = target[120:150]
    training_target = [training_target1, training_target2, training_target3]

    test = data
    solution = target
    confusion_matrix = np.zeros((3,3))

    testing_data1 = data[0:20]
    testing_data2 = data[50:70]
    testing_data3 = data[100:120]
    testing_data = [testing_data1,testing_data2,testing_data3]
    testing_target1 = target[0:20]
    testing_target2 = target[50:70]
    testing_target3 = target[100:120]
    testing_target = [testing_target1, testing_target2, testing_target3]

    #------------------Training-------------------
    training_ratio, confusion_training, W = iris.assignment_1_trainingset(training_data,training_target,3,confusion_matrix)
    print("Training:")
    print(training_ratio)
    print(confusion_training)

    #-------------------Testing---------------------
    test_ratio, confusion_test = iris.assignment_1_testingset(W, training_data, testing_data, testing_target, 3, confusion_matrix)
    print("Testing:")
    print(test_ratio)
    print(confusion_test)  

def main():
    print("First 30 for testing")
    task_a()
    print("-----------------------------------")
    print("Last 30")
    task_b()

main()