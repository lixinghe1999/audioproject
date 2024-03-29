
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import Dataset
from resnet18 import GE2ELoss, get_centroids, get_cossim, Swap, Mask
from torchvision import transforms
from sklearn.manifold import TSNE
import seaborn as sns
import random
class MyDataSet(Dataset):
    def __init__(self, path, ratio=1, augmentation=0):
        self.X = []
        self.Y = []
        self.augmentation = augmentation
        self.transform = transforms.Compose([Swap(30)])
        for i, p in enumerate(os.listdir(path)):
            person_path = os.path.join(path, p)
            files = os.listdir(person_path)
            if ratio > 0:
                files = files[: int(ratio * len(files))]
            else:
                files = files[int(ratio * len(files)):]
            for f in files:
                self.X.append(os.path.join(person_path, f))
                self.Y.append(int(i))
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):

        r = random.uniform(0, 1)
        if r < self.augmentation:
            return self.transform(np.load(self.X[index])), int(self.Y[index])
        else:
            return np.load(self.X[index]), int(self.Y[index])

class MyDataSet_Constrastive(Dataset):
    def __init__(self, path, shuffle=True, utter_num=6, ratio=1, augmentation=False):
        # data path
        self.X = []
        self.num_utterances = 0
        self.utter_num = utter_num
        for i, p in enumerate(os.listdir(path)):
            # iterate for each speaker
            person_path = os.path.join(path, p)
            files = os.listdir(person_path)
            if ratio > 0:
                files = files[: int(ratio * len(files))]
            else:
                files = files[int(ratio * len(files)):]
            num_batch = len(files) // self.utter_num
            for b in range(num_batch):
                utterances = []
                for f in files[b*self.utter_num: (b+1)*self.utter_num]:
                    utterances.append(os.path.join(person_path, f))
                self.X.append(utterances)
            self.num_utterances += num_batch

        self.path = path
        self.shuffle = shuffle
        self.transform = transforms.Compose([Swap(30)])
        self.augmentation = augmentation

    def __len__(self):
        return self.num_utterances
    def __getitem__(self, idx):
        utter_index = self.X[idx]
        utterance = []
        for i, file in enumerate(utter_index):
            data = np.load(file)
            if self.augmentation:
                data = self.transform(data)
            utterance.append(data)
        utterance = np.array(utterance)
        return utterance
class Experiment():
    def __init__(self, model, dataset, params, pretrain=None, single_modality=None):
        super(Experiment, self).__init__()
        self.single_modality = single_modality
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        #self.loss = GE2ELoss('cuda')
        if pretrain:
            self.model.load_state_dict(torch.load(pretrain))
        self.params = params
        self.device = torch.device('cuda')
        if len(dataset) == 2:
            train_dataset, test_dataset = dataset
        else:
            length = len(dataset)
            test_size = min(int(0.2 * length), 2000)
            train_size = length - test_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                        generator=torch.Generator().manual_seed(42))

        self.train_loader = Data.DataLoader(dataset=train_dataset, num_workers=4,
                                            batch_size=self.params['train_batch_size'], shuffle=True, drop_last=True)
        self.test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4,
                                           batch_size=self.params['test_batch_size'], shuffle=False, drop_last=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['LR'],
                                          weight_decay=self.params['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params['step_size'],
                                                         gamma=self.params['gamma'])

    def train(self):
        best_acc = 0.5
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
                self.optimizer.step()
            self.scheduler.step()
            acc = self.test()
            print(acc)
            acc_curve.append(acc)
            if acc > best_acc:
                best_acc = acc
                ckpt_best = self.model.state_dict()

        torch.save(ckpt_best, str(best_acc) + '_best.pth')
        plt.plot(acc_curve)
        plt.savefig(str(best_acc) + '_acc.png')
    def test(self):
        accuracy = []
        with torch.no_grad():
            for embeddings, cls in self.test_loader:
                embeddings = embeddings.to(device=self.device, dtype=torch.float)
                output = self.model(embeddings)
                cls_predict = torch.argmax(output, dim=-1).cpu()
                correct = cls_predict == cls
                ratio = torch.sum(correct) / len(correct)
                accuracy.append(ratio)
        acc = np.mean(accuracy)
        return acc
    def cluster_test(self):
        X = []
        Y = []
        with torch.no_grad():
            for embeddings, cls in self.test_loader:
                embeddings = embeddings.to(device=self.device, dtype=torch.float)
                output = self.model(embeddings, auth=True)
                output = output.cpu().numpy()
                cls = cls.cpu().numpy()
                X.append(output)
                Y.append(cls)
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)
        plt.figure()
        sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=Y, legend='full', palette=sns.color_palette("hls", 15))
        plt.show()
    def contrastive_test(self):
        batch_avg_EER = 0
        with torch.no_grad:
            for batch_id, embeddings in enumerate(self.test_loader):
                embeddings = embeddings.to(device=self.device, dtype=torch.float)
                assert self.params['num_utterances'] % 2 == 0

                enrollment_batch, verification_batch = torch.split(embeddings, int(embeddings.size(1) / 2), dim=1)
                enrollment_batch = torch.reshape(enrollment_batch, (
                    self.params['test_batch_size'] * self.params['num_utterances'] // 2,
                    enrollment_batch.size(2), enrollment_batch.size(3), enrollment_batch.size(4)))
                verification_batch = torch.reshape(verification_batch, (
                    self.params['test_batch_size'] * self.params['num_utterances'] // 2,
                    verification_batch.size(2), verification_batch.size(3), verification_batch.size(4)))

                perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
                unperm = list(perm)
                for i, j in enumerate(perm):
                    unperm[j] = i

                verification_batch = verification_batch[perm]
                enrollment_embeddings = self.model(enrollment_batch, auth=True)
                verification_embeddings = self.model(verification_batch, auth=True)

                verification_embeddings = verification_embeddings[unperm]

                enrollment_embeddings = torch.reshape(enrollment_embeddings,
                    (self.params['test_batch_size'], self.params['num_utterances'] // 2, enrollment_embeddings.size(1)))
                verification_embeddings = torch.reshape(verification_embeddings, (
                    self.params['test_batch_size'], self.params['num_utterances'] // 2, verification_embeddings.size(1)))

                enrollment_centroids = get_centroids(enrollment_embeddings)
                sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

                # calculating EER
                diff = 1;
                EER = 0;

                for thres in [0.01 * i + 0.5 for i in range(50)]:
                    sim_matrix_thresh = sim_matrix > thres

                    FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i
                                in range(self.params['test_batch_size'])])
                           / (self.params['test_batch_size'] - 1.0) / (float(self.params['num_utterances'] / 2)) / self.params['test_batch_size'])

                    FRR = (sum([self.params['num_utterances'] / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in
                                range(self.params['test_batch_size'])])
                           / (float(self.params['num_utterances']/ 2)) / self.params['test_batch_size'])

                    # Save threshold when FAR = FRR (=EER)
                    if diff > abs(FAR - FRR):
                        diff = abs(FAR - FRR)
                        EER = (FAR + FRR) / 2
                batch_avg_EER += EER
        batch_avg_EER = batch_avg_EER / (batch_id + 1)
        batch_avg_EER = batch_avg_EER.cpu().item()

        return batch_avg_EER
    def constrastive_train(self):
        EER_curve = []
        best_EER = 0.5
        for i in range(self.params['epoch']):
            for embeddings in tqdm(self.train_loader):
                embeddings = embeddings.to(device=self.device, dtype=torch.float)
                embeddings = torch.reshape(embeddings, (self.params['train_batch_size'] * self.params['num_utterances'], 2, 33, 151))
                perm = random.sample(range(0, self.params['train_batch_size'] * self.params['num_utterances']),
                                     self.params['train_batch_size'] * self.params['num_utterances'])
                unperm = list(perm)
                for i, j in enumerate(perm):
                    unperm[j] = i
                embeddings = embeddings[perm]
                # gradient accumulates
                self.optimizer.zero_grad()
                embeddings = self.model(embeddings)
                embeddings = embeddings[unperm]
                embeddings = torch.reshape(embeddings, (self.params['train_batch_size'], self.params['num_utterances'], -1))

                # get loss, call backward, step optimizer
                loss = self.loss(embeddings)  # wants (Speaker, Utterances, embedding)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            with torch.no_grad():
                EER = self.contrastive_test()
                print("\n average EER: %0.2f" % (EER))
            EER_curve.append(EER)
            if EER < best_EER:
                best_EER = EER
                ckpt_best = self.model.state_dict()
        self.scheduler.step()
        torch.save(ckpt_best, str(best_EER) + '_best.pth')
        plt.plot(EER_curve)
        plt.savefig(str(best_EER) + '_acc.png')


