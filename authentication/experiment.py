
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data

from tqdm import tqdm
import numpy as np
import os


class MyDataSet:
    def __init__(self, path):
        self.files = os.listdir(path)
        self.X = []
        self.Y = []
        for i, f in enumerate(self.files):
            x = np.load(os.path.join(path, f))
            self.X.append(x)
            N = np.shape(x)[0]
            self.Y.append(np.ones(N) * i)
        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y)

    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return self.X[index], int(self.Y[index])

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
    elif isinstance(m, torch.nn.Conv1d):
        m.weight.data.normal_(0, 0.01)

class Experiment():
    def __init__(self,
                 model, dataset, params, pretrain=None, single_modality=None):
        super(Experiment, self).__init__()
        self.single_modality = single_modality
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        if pretrain:
            self.model.load_state_dict(torch.load(pretrain))
        else:
            self.model.apply(init_weights)
        self.params = params
        self.device = torch.device('cuda')
        if len(dataset)==2:
            train_dataset, test_dataset = dataset
        else:
            length = len(dataset)
            test_size = min(int(0.2 * length), 2000)
            train_size = length - test_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
        self.train_loader = Data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=self.params['batch_size'], shuffle=True)
        self.test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=self.params['batch_size'], shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['LR'],weight_decay=self.params['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def train(self):
        best_loss = 0.5
        acc_curve = []
        for i in range(self.params['epoch']):
            for embeddings, cls in tqdm(self.train_loader):
                embeddings = embeddings.to(device=self.device, dtype=torch.float)
                cls = torch.nn.functional.one_hot(cls, num_classes=15)
                cls = cls.to(device=self.device, dtype=torch.float)
                output = self.model(embeddings)
                loss = self.loss(output, cls)
                self.optimizer.zero_grad()

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
            accuracy = []
            with torch.no_grad():
                for embeddings, cls in tqdm(self.train_loader):
                    embeddings = embeddings.to(device=self.device, dtype=torch.float)
                    output = self.model(embeddings)
                    cls_predict = torch.argmax(output, dim=-1).cpu()
                    correct = cls_predict == cls
                    ratio = torch.sum(correct) / len(correct)
                    accuracy.append(ratio)
                acc = np.mean(accuracy)
                print(acc)
                acc_curve.append(acc)
                if acc < best_loss:
                    best_loss = acc
                    ckpt_best = self.model.state_dict()
            self.scheduler.step()
        torch.save(ckpt_best, 'best.pth')
        plt.plot(acc_curve)
        plt.savefig('acc.png')

    def evaluation(self):
        a1, a2 = [], []
        with torch.no_grad():
            for (imu, audio, fls) in self.test_loader:
                output, fls = self.sample(imu, audio, fls, inference=True)
                output = output.cpu().numpy()
                fls = fls.cpu().numpy()
                a1.append(output)
                a2.append(fls)
        dict = self.area_loss(np.concatenate(a1, axis=0), np.concatenate(a2, axis=0))
        return dict
