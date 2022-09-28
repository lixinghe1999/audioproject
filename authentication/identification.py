
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
from sklearn.svm import SVC, OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import argparse
from experiment import Experiment, MyDataSet, MyDataSet_Constrastive
from model import ResNet18
import yaml
import json
from torch.utils.data import ConcatDataset


def identification(X, Y, ratio=0.1):
    random_number = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, stratify=Y, random_state=random_number)
    #clf = SVC(kernel='rbf', class_weight='balanced').fit(X_train, y_train)
    clf = MLPClassifier(hidden_layer_sizes=(128, 128, 128), max_iter=500).fit(X_train, y_train)
    return y_test, clf.predict(X_test), clf

def Mel_split(X, Y, ratio=0.2):
    random_number = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, stratify=Y, random_state=random_number)

    results = []
    for k in range(12):
        X_clip = np.concatenate([X_train[:, k * 10 : (k + 1) * 10], X_train[:, k * 10 : (k + 1) * 10]], axis=1)
        clf = SVC(kernel='rbf', class_weight='balanced').fit(X_clip, y_train)
        pred = clf.predict(np.concatenate([X_test[:, k * 10 : (k + 1) * 10], X_test[:, k * 10 : (k + 1) * 10]], axis=1))
        results.append(pred)
    results = np.array(results)
    axis = 0
    u, indices = np.unique(results, return_inverse=True)
    hard_vote = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(results.shape), None, np.max(indices) + 1), axis=axis)]
    return y_test, hard_vote

def load_data(path):
    people = os.listdir(path)
    X = []
    Y = []
    for i, p in enumerate(people):
        x = np.load(os.path.join(path, p))
        X.append(x)
        N = np.shape(x)[0]
        Y.append(np.ones(N) * i)
    return np.concatenate(X, axis=0), np.concatenate(Y)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=0, required=False)
    # mode 0: utterance-level embeddings
    # mode 1: deep learning
    # mode 2: phone-level embeddings
    args = parser.parse_args()
    # palette = sns.color_palette("bright", 17)
    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X2)
    # sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=Y, legend='full', palette=palette)
    # plt.show()
    if args.mode == 0:
        #X1, Y = load_data('speaker_embedding/DNN_embedding')
        X2, Y = load_data('speaker_embedding/mfcc_embedding')
        X3, Y = load_data('speaker_embedding/bcf_embedding')

        #X = np.concatenate([X2, X3], axis=1)
        gt, p1, clf1 = identification(X3, Y)
        #gt, p2 = Mel_split(X3, Y)
        print(balanced_accuracy_score(gt, p1))
        mat = confusion_matrix(gt, p1, normalize='true')
        plt.imshow(mat)
        plt.colorbar()
        plt.show()
    elif args.mode == 1:
        with open('model.yaml', 'r') as file:
            config = yaml.safe_load(file)
        model = ResNet18(output_dim=15).cuda()
        # train_clean_dataset = MyDataSet_Constrastive('speaker_embedding/DNN_embedding',
        #                                  utter_num=config['exp_params']['num_utterances'], ratio=0.8, augmentation=False)
        # train_noisy_dataset = MyDataSet_Constrastive('speaker_embedding/noise_DNN_embedding',
        #                                              utter_num=config['exp_params']['num_utterances'],
        #                                              ratio=0.8, augmentation=False)
        # train_dataset = ConcatDataset([train_clean_dataset, train_noisy_dataset])
        #
        # test_clean_dataset = MyDataSet_Constrastive('speaker_embedding/DNN_embedding',
        #                                              utter_num=config['exp_params']['num_utterances'], ratio=-0.2)
        # test_noisy_dataset = MyDataSet_Constrastive('speaker_embedding/noise_DNN_embedding',
        #                                              utter_num=config['exp_params']['num_utterances'], ratio=-0.2)
        # test_dataset = ConcatDataset([test_clean_dataset, test_noisy_dataset])

        # train_dataset = MyDataSet('speaker_embedding/DNN_embedding', ratio=0.8)
        # test_dataset = MyDataSet('speaker_embedding/DNN_embedding', ratio=-0.2)
        dataset = MyDataSet('speaker_embedding/DNN_embedding')
        noisy_dataset = MyDataSet('speaker_embedding/noise_DNN_embedding')
        dataset = ConcatDataset([dataset, noisy_dataset])
        Exp = Experiment(model, dataset, config['exp_params'], pretrain='97_noisy.pth')
        #Exp.constrastive_train()
        #EER = Exp.contrastive_test()
        #print(EER)
        Exp.cluster_test()
    else:
        path = 'speaker_embedding/phone_embedding'
        people = os.listdir(path)
        X_train = {}
        Y_train = {}
        X_test = []
        Y_test = []
        classifiers = {}
        for i, p in enumerate(people):
            with open(os.path.join(path, p), 'r') as f:
                data = json.load(f)
            # keys = ['ʊ', 'm', 'ɔ', 'n', 'ɪ', 'f', 'ɛ', 'ə', 'v', 'a', 's', 't', 'o', 'd', 'i', 'k', 'ɡ', 'æ', 'ɒ', 'w', 'e',
            #  'ɹ', 'iː', 'ŋ', 'b', 'ʌ', 'p', 'r', 'l', 'u', 'tʰ', 'x', 'j', 'uː',
            #  'h', 'ɑ', 'ʃ', 'ð', 'ɻ', 'z', 'θ', 'ɯ', 'pʰ', 'ʔ', 'ʒ']
            print(len(data))
            N = len(data)
            N_train = int(0.8 * N)
            for j, dict in enumerate(data):
                if j < N_train:
                    for key in dict:
                        for d in dict[key]:
                            pad = 720 - len(d[0])
                            x = d[0] + [0] * pad
                            x = np.abs(np.fft.fft(x))[:360:10]
                            x1 = np.linalg.norm(np.abs(np.fft.fft(d[1]))[:36], axis=1)/x
                            x2 = np.linalg.norm(np.abs(np.fft.fft(d[2]))[:36], axis=1)/x
                            x = np.concatenate([x1, x2])
                            if key not in X_train:
                                X_train[key] = [x]
                                Y_train[key] = [i]
                            else:
                                X_train[key].append(x)
                                Y_train[key].append(i)
                else:
                    X_test.append(dict)
                    Y_test.append(i)
        # training
        for key in X_train:
            X = X_train[key]
            Y = Y_train[key]
            if len(set(Y)) == 1:
                continue
            X = np.stack(X, axis=0)
            Y = np.array(Y)
            #clf = SVC(kernel='rbf', class_weight='balanced').fit(X, Y)
            clf = MLPClassifier(hidden_layer_sizes=(128, 128, 128), max_iter=500).fit(X, Y)
            classifiers[key] = clf
        # testing
        predictions = []
        for dict_X, Y in zip(X_test, Y_test):
            phones = []
            for key in dict_X:
                if key in classifiers:
                    clf = classifiers[key]
                    for d in dict_X[key]:
                        pad = 720 - len(d[0])
                        x = d[0] + [0] * pad
                        x = np.abs(np.fft.fft(x))[:360:10]
                        x1 = np.linalg.norm(np.abs(np.fft.fft(d[1]))[:36], axis=1) / x
                        x2 = np.linalg.norm(np.abs(np.fft.fft(d[2]))[:36], axis=1) / x
                        x = np.concatenate([x1, x2])
                        pred = clf.predict(x.reshape(1, -1))
                        phones.append(pred[0])
            phones = np.array(phones)
            if len(phones) > 1:
                u, indices = np.unique(phones, return_inverse=True)
                hard_vote = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(phones.shape), None,
                                                np.max(indices) + 1), axis=0)]
            else:
                hard_vote = phones
            predictions.append(hard_vote)

        print(balanced_accuracy_score(Y_test, predictions))
        fig, axs = plt.subplots(2)
        mat = confusion_matrix(Y_test, predictions, normalize='true')
        axs[0].imshow(mat)

        path = 'speaker_embedding/bcf_embedding'
        people = os.listdir(path)
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        for i, p in enumerate(people):
            x = np.load(os.path.join(path, p))
            N = np.shape(x)[0]
            N_train = int(0.8 * N)
            X_train.append(x[:N_train, :])
            Y_train.append(np.ones(N_train) * i)
            X_test.append(x[N_train:, :])
            Y_test.append(np.ones(N - N_train) * i)
        X_train = np.concatenate(X_train, axis=0)
        Y_train = np.concatenate(Y_train)
        X_test = np.concatenate(X_test, axis=0)
        Y_test = np.concatenate(Y_test)
        clf = SVC(kernel='rbf', class_weight='balanced').fit(X_train, Y_train)
        bcf_pred = clf.predict(X_test)
        print(balanced_accuracy_score(Y_test, bcf_pred))
        mat = confusion_matrix(Y_test, bcf_pred, normalize='true')
        axs[1].imshow(mat)
        plt.show()



