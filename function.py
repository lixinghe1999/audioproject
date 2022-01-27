import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
function_length = 33
if __name__ == "__main__":

    functions = os.listdir('transfer_function')
    d = {}
    for f in functions:
        npzs = np.load('transfer_function/' + f)
        name = f.split('_')[1]
        if name not in d:
            d[name] = {}
            d[name]['r'] = []
            d[name]['v'] = []
        d[name]['r'].append(npzs['response'])
        d[name]['v'].append(npzs['variance'])

    iterated = {}
    max_iteration = 5
    X = []
    Y = []
    candidate = ["liang", "wu", "he", "hou", "zhao", "shi"]
    color_map = {"liang":'b', "wu":'g', "he":'r', "hou":'y', "zhao":'k', "shi":'w'}
    fig, ax = plt.subplots(1, sharex=True, figsize=(5, 4))
    for i in range(len(candidate)):
        name = candidate[i]
        response = d[name]['r']
        variance = d[name]['v']
        iterated_response, iterated_variance = np.zeros((function_length)), np.zeros((function_length))
        for j in range(len(response)):
            select = response[j] > 0
            iterated_response[select] = 0.5 * response[j][select] + 0.5 * iterated_response[select]
            iterated_variance[select] = 0.5 * variance[j][select] + 0.5 * iterated_variance[select]
            if (j + 1) % max_iteration == 0:
                X.append(iterated_response)
                Y.append(i)
                #plt.plot(np.arange(0, 801, 25), iterated_variance)
                iterated_response, iterated_variance = np.zeros((function_length)), np.zeros((function_length))
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel(r'Ratio ($S_{Acc}/S_{Mic}$)')
    # plt.savefig('transfer.eps')
    # plt.show()

    X = np.array(X)
    Y = np.array(Y)
    clf = SVC(decision_function_shape='ovr')
    print(X.shape, Y.shape)
    clf.fit(X, Y)
    print(accuracy_score(Y, clf.predict(X)))
