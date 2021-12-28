import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats.stats import pearsonr
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
            d[name]['n'] = []
        d[name]['r'].append(npzs['response'])
        d[name]['v'].append(npzs['variance'])
        d[name]['n'].append(npzs['noise'])

    iterated = {}
    max_iteration = 1
    X = []
    Y = []
    candidate = ["liang", "shuai", "shen", "wu", "he", "hou", "zhao", "shi"]
    for i in range(len(candidate)):
        name = candidate[i]
        response = d[name]['r']
        iterated = np.zeros((112))
        for j in range(len(response)):
            select = response[j] > 0
            iterated[select] = 0.1 * response[j][select] + 0.9 * iterated[select]
            if (j + 1) % max_iteration == 0:
                X.append(iterated)
                Y.append(i)
                iterated = np.zeros((112))
    clf = SVC(decision_function_shape='ovr')
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    clf.fit(X, Y)
    print(accuracy_score(Y, clf.predict(X)))


    # corr = np.zeros((len(candidate), len(candidate)))
    # for i in range(len(candidate)):
    #     for j in range(len(candidate)):
    #         i_j_corr = np.corrcoef(np.hstack((iterated[candidate[i]], iterated[candidate[j]])), rowvar=False)
    #         l1, l2 = np.shape(iterated[candidate[i]])[1], np.shape(iterated[candidate[j]])[1]
    #         true_corr = i_j_corr[:l1, l2:]
    #         flat = list(set(i_j_corr.flatten()))
    #         flat.sort()
    #         corr[i, j] = np.mean(flat)
    # plt.imshow(corr)
    # plt.colorbar()
    # plt.show()