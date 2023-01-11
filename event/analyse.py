import torch
from model import AudioCLIP
from utils.datasets.esc50 import ESC50
from utils.datasets.split_dataset import split_dataset, split_dataset_type, split_type
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
def data_preprocess(text_features):
    data = np.load('user_specific.npz', allow_pickle=True)
    perf = data['metric'][:, 0]
    type_list = data['type_list']
    var = []
    for i, user_type in enumerate(type_list):
        user_features = text_features[0, user_type[3:]]
        user_var = np.mean(np.var(user_features, axis=0))
        var.append(user_var)
    var = np.array(var)
    return perf, var
def vis(perf, var):
    plt.scatter(var, perf)
    m, b = np.polyfit(var, perf, 1)
    plt.plot(var, m * var + b)
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
    perf, var = data_preprocess(text_features)
    vis(perf, var)

    # palette = sns.color_palette("bright", 10)
    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(text_features[0])
    # sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=cluster, legend='full', palette=palette)
    # plt.show()

