import torch
from model import AudioCLIP
from utils.datasets.esc50 import ESC50
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
def cal_feature(user_features, type='var'):
    if type == 'var':
        feature = np.mean(np.var(user_features[3:], axis=0))
    elif type == 'sim':
        feature = user_features[3:] @ user_features[:3].transpose(-1, -2)
        feature = [np.mean(feature), np.mean(np.max(feature, axis=1)), np.mean(np.min(feature, axis=1))]
    return feature
def data_preprocess(text_features):
    data = np.load('user_specific.npz', allow_pickle=True)
    perf = data['metric'][:, 0]
    type_list = data['type_list']
    features = []
    for i, user_type in enumerate(type_list):
        feature = cal_feature(text_features[0, user_type], type='sim')
        features.append(feature)
    features = np.array(features)
    return perf, features
def vis(perf, features):
    poly_reg = PolynomialFeatures(degree=1)
    x_poly = poly_reg.fit_transform(features)
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(x_poly, perf)
    predict_perf = lin_reg.predict(x_poly)
    error = np.mean(np.abs(predict_perf - perf))
    print(error)
    plt.plot(predict_perf)
    plt.plot(perf)
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
    vis(perf, features)

    # palette = sns.color_palette("bright", 10)
    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(text_features[0])
    # sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=cluster, legend='full', palette=palette)
    # plt.show()

