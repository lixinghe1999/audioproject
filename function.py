import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

function_length = 129
if __name__ == "__main__":

    functions = os.listdir('transfer_function')
    num = len(functions)
    ratio = 0.2
    X1 = np.empty((num, function_length))
    X2 = np.empty((num, function_length))
    Y = np.empty((num,))
    candidate = {'yan':1, 'he':2, 'hou':3, 'shi':4, 'shuai':5, 'wu':6, 'liang':7, '1':8, '2':9, '3':10, '4':11, '5':12, '6':13, '7':14, '8':15}
    for i, f in enumerate(functions):
        npzs = np.load('transfer_function/' + f)
        name = f.split('_')[1]
        index = candidate[name]
        m = np.max(npzs['response'])
        if m > 0:
            X1[i, :] = npzs['response']/m
            X2[i, :] = npzs['variance']/m
            Y[i] = index
        else:
            X1[i, :] = npzs['response']
            X2[i, :] = npzs['variance']
            Y[i] = index


    X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=ratio, stratify=Y, random_state=1)
    print(X1.shape)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    print(accuracy_score(y_test, clf.predict(X_test)))
