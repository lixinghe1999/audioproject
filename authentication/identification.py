from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import torch


def get_spk_emb(audio_file_dir, segment_len=960000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resemblyzer_encoder = VoiceEncoder(device=device, verbose=False)

    wav = preprocess_wav(audio_file_dir)
    l = len(wav) // segment_len # segment_len = 16000 * 60
    l = np.max([1, l])
    all_embeds = []
    for i in range(l):
        mean_embeds, cont_embeds, wav_splits = resemblyzer_encoder.embed_utterance(
            wav[segment_len * i:segment_len* (i + 1)], return_partials=True, rate=2)
        all_embeds.append(mean_embeds)
    all_embeds = np.array(all_embeds)
    mean_embed = np.mean(all_embeds, axis=0)

    return mean_embed, all_embeds
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import argparse


def identification(X, Y, ratio=0.5, state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, stratify=Y, random_state=state)
    clf = SVC(kernel='rbf', class_weight='balanced')
    clf.fit(X_train, y_train)
    return y_test, clf.predict(X_test), clf

def authentication(X):
    clf = OneClassSVM(kernel='rbf', gamma='auto')
    clf.fit(X)
    return clf

def load_data(file, save=None):
    function_length = 129
    n_mfcc = 20
    functions = os.listdir(file)
    num = len(functions)
    X1 = np.empty((num, function_length))
    X2 = np.empty((num, function_length))
    X3 = np.empty((num, n_mfcc))
    Y = np.empty((num,))
    for i, f in enumerate(functions):
        npzs = np.load(os.path.join(file, f))
        index = int(f[:-4].split('_')[0])
        r1 = npzs['r1']
        r2 = npzs['r2']
        m1 = np.max(r1)
        m2 = np.max(r2)
        X1[i, :] = r1 / m1 if m1 > 0 else r1
        X2[i, :] = r2 / m2 if m2 > 0 else r2
        X3[i, :] = npzs['mfcc']
        if save is not None:
            if index == save:
                Y[i] = 0
            else:
                Y[i] = 1
        else:
            Y[i] = index
    return X1, X2, X3, Y
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=0, required=False)
    args = parser.parse_args()
    # palette = sns.color_palette("bright", 17)
    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X2)
    # sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=Y, legend='full', palette=palette)
    # plt.show()
    if args.mode == 0:

        X1, X2, X3, Y = load_data('transfer_function')
        random_number = np.random.randint(0, 100)
        gt, p1, clf1 = identification(X1, Y, state=random_number)
        _, p2, clf2 = identification(X2, Y, state=random_number)
        _, p3, clf3 = identification(X3, Y, state=random_number)
        P = np.vstack([p1, p2, p3])
        u, indices = np.unique(P, return_inverse=True)
        vote = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(P.shape), None, np.max(indices) + 1), axis=0)]
        print(balanced_accuracy_score(gt, p1), balanced_accuracy_score(gt, p2), balanced_accuracy_score(gt, p3))
        print(balanced_accuracy_score(gt, vote))
        mat = confusion_matrix(gt, vote, normalize='true')
        plt.imshow(mat)
        plt.colorbar()
        plt.show()
    elif args.mode == 1:
        random_number = np.random.randint(0, 100)
        for i in range(16):
            X1, X2, X3, Y = load_data('transfer_function', save=i)
            gt, p1, clf1 = identification(X1, Y, state=random_number)
            _, p2, clf2 = identification(X2, Y, state=random_number)
            _, p3, clf3 = identification(X3, Y, state=random_number)

            P = np.vstack([p1, p2, p3])
            u, indices = np.unique(P, return_inverse=True)
            vote = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(P.shape), None, np.max(indices) + 1), axis=0)]
            #print(balanced_accuracy_score(gt, p1), balanced_accuracy_score(gt, p2), balanced_accuracy_score(gt, p3))
            print(balanced_accuracy_score(gt, vote))

    else:
        pair = [0, 3, 2, 6]
        #pair = [0, 2]
        random_number = np.random.randint(0, 100)
        for i in range(len(pair)):
            X1, X2, X3, Y = load_data('transfer_function', save=pair[i])

            gt, _, clf1 = identification(X1, Y, state=random_number)
            _, _, clf2 = identification(X2, Y, state=random_number)
            _, _, clf3 = identification(X3, Y, state=random_number)
            X1, X2, X3, Y = load_data('attack_transfer_function/human', save=i)
            select = Y == 0
            X1 = X1[select]
            X2 = X2[select]
            X3 = X3[select]
            Y = Y[select]
            # for x, y in zip(X1, Y):
            #     print(clf1.predict([x]), y)
            #     plt.plot(x)
            #     plt.show()
            p1 = clf1.predict(X1)
            p2 = clf2.predict(X2)
            p3 = clf3.predict(X3)
            P = np.vstack([p1, p2, p3])
            u, indices = np.unique(P, return_inverse=True)
            vote = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(P.shape), None, np.max(indices) + 1), axis=0)]
            print(sum(p1 == 0)/len(Y), sum(p2 == 0)/len(Y), sum(p3 == 0)/len(Y))
            print(sum(vote == 0)/len(Y))

            # mat = confusion_matrix(Y, vote, normalize='true')
            # plt.imshow(mat)
            # plt.colorbar()
            # plt.show()


