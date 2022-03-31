import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns

def fitting(X, Y, ratio=0.2, state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, stratify=Y, random_state=state)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return y_test, clf.predict(X_test)


if __name__ == "__main__":
    palette = sns.color_palette("bright", 17)
    function_length = 129
    n_mfcc = 20
    functions = os.listdir('transfer_function')
    num = len(functions)
    ratio = 0.8
    X1 = np.empty((num, function_length))
    X2 = np.empty((num, function_length))
    X3 = np.empty((num, n_mfcc))
    Y = np.empty((num,))
    pre_index = 0
    for i, f in enumerate(functions):
        npzs = np.load('transfer_function/' + f)
        index = int(f[:-4].split('_')[0])
        r1 = npzs['r1']
        r2 = npzs['r2']
        m1 = np.max(r1)
        m2 = np.max(r2)
        X1[i, :] = r1 / m1 if m1 > 0 else r1
        X2[i, :] = r2 / m2 if m2 > 0 else r2
        X3[i, :] = npzs['mfcc']
        Y[i] = index

    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X2)
    # sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=Y, legend='full', palette=palette)
    # plt.show()
    random_number = np.random.randint(0, 100)
    gt, p1 = fitting(X1, Y, state=random_number)
    _, p2 = fitting(X2, Y, state=random_number)
    _, p3 = fitting(X3, Y, state=random_number)
    P = np.vstack([p1, p2, p3])
    u, indices = np.unique(P, return_inverse=True)
    vote = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(P.shape), None, np.max(indices) + 1), axis=0)]
    print(accuracy_score(gt, p1))
    print(accuracy_score(gt, p2))
    print(accuracy_score(gt, p3))
    print(accuracy_score(gt, vote))
    mat = confusion_matrix(gt, vote, normalize='true')
    plt.imshow(mat)
    plt.colorbar()
    plt.show()

