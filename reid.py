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
    max_iteration = 5
    X = []
    Y = []
    candidate = ["liang", "shuai", "shen", "wu", "he", "hou", "zhao", "shi"]
    #candidate = ["zhao"]
    for i in range(len(candidate)):
        count = 0
        name = candidate[i]
        response = d[name]['r']
        variance = d[name]['v']
        iterated_response = np.zeros((112))
        iterated_variance = np.zeros((112))
        for j in range(len(response)):
            select = response[j] > 0
            iterated_response[select] = 0.5 * response[j][select] + 0.5 * iterated_response[select]
            iterated_variance[select] = 0.5 * variance[j][select] + 0.5 * iterated_variance[select]
            if (j + 1) % max_iteration == 0:
                X.append(iterated_response)
                Y.append(i)
                np.savez('iterated_function/' + str(count) + '_' + name + '_transfer_function.npz',
                         response=iterated_response, variance=iterated_variance)
                iterated_response = np.zeros((112))
                iterated_variance = np.zeros((112))

                count += 1
    X = np.array(X)
    Y = np.array(Y)
    clf = SVC(decision_function_shape='ovr')
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