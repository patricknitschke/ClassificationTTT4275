#Decision tree classifier

import numpy as np

from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25)


from sklearn import tree
classifier = tree.DecisionTreeClassifier()

classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, prediction))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, prediction))