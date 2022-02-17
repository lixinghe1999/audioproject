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
    candidate = ['yan', 'he', 'hou', 'shi', 'shuai', 'wu', 'liang', "1", "2", "3", "4", "5", "6", "7", "8"]
    X1 = []
    X2 = []
    Y = []
    fig, ax = plt.subplots(1, sharex=True, figsize=(5, 4))
    for i in range(len(candidate)):
        name = candidate[i]
        response = d[name]['r']
        variance = d[name]['v']
        for j in range(len(response)):
            normalization = np.max(response[j])
            X1.append(response[j]/normalization)
            X2.append(variance[j]/normalization)
            Y.append(i)
        # X1 = np.mean(X1, axis=0)
        # X2 = np.mean(X2, axis=0)
    #     plt.errorbar(np.arange(0, 801, 25), X1, X2, fmt='-o', label=candidate[i], capsize=4)
    # plt.legend()
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel(r'Ratio ($S_{Acc}/S_{Mic}$)')
    # plt.savefig('transfer_variance.eps')
    # plt.show()

    X1 = np.array(X1)
    Y = np.array(Y)
    clf = SVC(decision_function_shape='ovr')
    print(X1.shape, Y.shape)
    clf.fit(X1, Y)
    print(accuracy_score(Y, clf.predict(X1)))
