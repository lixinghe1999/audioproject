'''
Implement the method to get relative clean result
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.metrics import balanced_accuracy_score
def dot_plot(cosine, y, correct_cosine):
    check = np.argmax(cosine, axis=-1) == np.argmax(y, axis=-1)
    print(np.sum(check)/total, np.sum(check) / np.shape(check)[0])
    plt.scatter(np.arange(0, y.shape[0])[check], correct_cosine[check], c='r')
    plt.scatter(np.arange(0, y.shape[0])[~check], correct_cosine[~check], c='b')
    plt.show()


if __name__ == "__main__":
    embed = np.load('save_embedding.npz')
    audio = embed['audio']
    image = embed['image']
    text = embed['text']
    y = embed['y']
    total = y.shape[0]
    print(audio.shape, image.shape, text.shape, y.shape)
    cosine = audio @ text[0].transpose()
    correct_cosine = np.sum(cosine * y, axis=-1)

    # without selection
    # dot_plot(cosine, y, correct_cosine)

    # select by the threshold
    # above_threshold = correct_cosine > 0.15
    # dot_plot(cosine[above_threshold], y[above_threshold], correct_cosine[above_threshold])

    # select by skewness
    sort_cosine = np.sort(cosine, axis=-1)
    top_ratio = sort_cosine[:, -1] / sort_cosine[:, -2]
    gap = np.mean(sort_cosine[:, -3:], axis=-1)
    above_threshold = top_ratio > 1.1
    dot_plot(cosine[above_threshold], y[above_threshold], correct_cosine[above_threshold])

    # features = np.stack([top_ratio], axis=1)
    # cls = np.argmax(cosine, axis=-1) == np.argmax(y, axis=-1)
    # clf = SVC(kernel='linear', class_weight='balanced').fit(features, cls)
    # above_threshold = clf.predict(features)
    # print(balanced_accuracy_score(above_threshold, cls))
    # dot_plot(cosine[above_threshold], y[above_threshold], correct_cosine[above_threshold])

