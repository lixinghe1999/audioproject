'''
Implement the method to get relative clean result
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
def pseduo_label(embeddings, text, y, method='skewness'):
    total = len(y)
    cosine = embeddings @ text.transpose()
    correct_cosine = cosine[np.arange(total), y]
    zero_shot = np.argmax(cosine, axis=-1) == y
    print('zero-shot performance:', sum(zero_shot)/total)
    sort_cosine = np.sort(cosine, axis=-1)
    top_cos = sort_cosine[:, -1]
    top_ratio = sort_cosine[:, -1] / sort_cosine[:, -2]
    gap = np.mean(sort_cosine[:, -3:], axis=-1) - np.mean(sort_cosine[:, :-3], axis=-1)
    mean = np.mean(sort_cosine, axis=-1)
    var = np.var(sort_cosine, axis=-1)

    if method == 'threshold':
        above_threshold = top_cos > 0.2
        dot_plot(cosine[above_threshold], y[above_threshold], correct_cosine[above_threshold], total)
    elif method == 'skewness':
        above_threshold = top_ratio > 1.1
        dot_plot(cosine[above_threshold], y[above_threshold], correct_cosine[above_threshold], total)
    else:
        above_threshold = gap > 0.1
        dot_plot(cosine[above_threshold], y[above_threshold], correct_cosine[above_threshold], total)
    label = np.argmax(cosine, axis=-1)
    return above_threshold, label
        # features = np.stack([top_cos, top_ratio, gap, mean, var], axis=1)
        # cls = np.argmax(cosine, axis=-1) == y
        # X_train, X_test, y_train, y_test = train_test_split(features, cls, test_size=0.2, random_state=42)
        #
        # clf = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
        # above_threshold = clf.predict(X_test)
        # print(balanced_accuracy_score(above_threshold, y_test))
        # dot_plot(cosine[above_threshold], y[above_threshold], correct_cosine[above_threshold])

def dot_plot(cosine, y, correct_cosine, total):
    check = np.argmax(cosine, axis=-1) == y
    print('utilized percentage:', np.sum(check)/total, 'accuracy:', np.sum(check) / np.shape(check)[0])
    # plt.scatter(np.arange(0, y.shape[0])[check], correct_cosine[check], c='r')
    # plt.scatter(np.arange(0, y.shape[0])[~check], correct_cosine[~check], c='b')
    # plt.show()




