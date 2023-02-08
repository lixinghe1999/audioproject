from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.AVnet import AVnet
from utils.train import zero_shot_eval, eval_step
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
# remove annoying librosa warning
def train(train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=16, shuffle=True,
                                               drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=16, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    loss = torch.nn.CrossEntropyLoss()
    for e in range(10):
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            audio, image, text, files = batch
            print(audio.shape, image.shape)
            predict = model(audio.to(device), image.to(device))
            y = torch.zeros(len(text), len(dataset.class_idx_to_label), dtype=torch.int8)
            for item_idx, label in enumerate(text):
                class_ids = dataset.label_to_class_idx[label]
                y[item_idx][class_ids] = 1
            l = loss(predict, y.to(device))
            l.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        acc = []
        with torch.no_grad():
            for batch in test_loader:
                data, _, label = batch
                data = data.to(device)
                predict = model(data)
                acc.append((torch.argmax(predict, dim=-1).cpu() == label).sum() / len(label))
        print('epoch', e, np.mean(acc))
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    model = AVnet().to(device)
    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    train(train_dataset, test_dataset)

