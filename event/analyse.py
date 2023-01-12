import torch
from model import AudioCLIP
from utils.datasets.esc50 import ESC50
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.svm import SVR, SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
def cal_feature(text_features, type='var'):
    if type == 'var':
        feature = np.mean(np.var(text_features[:-1], axis=0))
    elif type == 'sim':
        feature = text_features[-1, :] @ text_features[:-1].transpose(-1, -2)
        feature = np.sort(feature)
        # feature = [np.mean(feature), np.var(feature)]
    return feature
def data_preprocess(text_features):
    data = np.load('user_specific.npz', allow_pickle=True)
    perf = data['metric'][:, 0]
    # perf = perf.reshape(-1, 10)
    # perf /= np.min(perf, axis=1, keepdims=True)
    # perf = perf.reshape(-1)
    type_list = data['type_list']
    features = []
    for i, user_type in enumerate(type_list):
        feature = cal_feature(text_features[0, user_type], type='sim')
        features.append(feature)
    features = np.array(features)
    return perf, features
def vis(perf, features):
    perf = perf.reshape(-1, 10)
    success_index = np.argmax(perf, axis=1)
    success_index += np.arange(0, 500, 10)
    plt.scatter(features[:, 0], features[:, 1], c='b')
    plt.scatter(features[success_index, 0], features[success_index, 1], c='r')
    plt.show()
def fit(perf, features):
    print(features.shape)
    perf = perf.reshape(-1, 10)
    sort_index = np.argsort(perf, axis=1) + np.arange(0, 500, 10).reshape(-1, 1)
    high_index = sort_index[:, :3].reshape(-1)
    middle_index = sort_index[:, 3:6].reshape(-1)
    cls = np.ones(len(perf), dtype=int)
    cls[middle_index] = 2
    cls[high_index] = 3

    clf = SVC(kernel='rbf', class_weight='balanced').fit(features, cls)
    #clf = MLPClassifier(hidden_layer_sizes=(32, 32, 16), max_iter=300).fit(features, cls)
    predict_perf = clf.predict(features)

    # poly_reg = PolynomialFeatures(degree=1)
    # x_poly = poly_reg.fit_transform(features)
    # lin_reg = linear_model.LinearRegression()
    # lin_reg.fit(x_poly, perf)
    # predict_perf = lin_reg.predict(x_poly)

    error = balanced_accuracy_score(predict_perf, cls)
    print(error)
    plt.plot(predict_perf)
    plt.plot(cls)
    # m, b = np.polyfit(features, perf, 1)
    # plt.plot(features, m * features + b)
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    SAMPLE_RATE = 44100

    train_dataset = ESC50('../dataset/ESC50', fold=1, train=True, sample_rate=SAMPLE_RATE, few_shot=None)
    test_dataset = ESC50('../dataset/ESC50', fold=1, train=False, sample_rate=SAMPLE_RATE)

    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    model.eval()
    with torch.no_grad():
        ((_, _, text_features), _), _ = model(text=[
            [test_dataset.class_idx_to_label[class_idx]]
            for class_idx in sorted(test_dataset.class_idx_to_label.keys())
        ], batch_indices=torch.arange(len(test_dataset.class_idx_to_label), dtype=torch.int64, device=device))
        text_features = text_features.unsqueeze(1).transpose(0, 1).detach().cpu().numpy()
    perf, features = data_preprocess(text_features)
    fit(perf, features)
    # vis(perf, features)
    # palette = sns.color_palette("bright", 10)
    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(text_features[0])
    # sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=cluster, legend='full', palette=palette)
    # plt.show()

