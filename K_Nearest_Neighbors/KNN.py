# @time    : 2017/12/3 14:45
# @Author  : chiew
# @File    : KNN.py

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# print(iris_X[:2, :])
# print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, random_state=4)

# print(y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)
print(knn.score(X_test, y_test))
