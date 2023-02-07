from torchvision.models import resnet18
import numpy as np
from pseudo import pseduo_label
import torch.utils.data as td
import torchvision as tv
import ffmpeg
import os
import librosa
import torch
from tqdm import tqdm
import random
from collections import defaultdict
loss = torch.nn.CrossEntropyLoss()
class pseudo_dataset(td.Dataset):
    def __init__(self,
                 root,
                 file_list,
                 pseduo_label,
                 label,
                 sample_rate: int = 44100,
                 transform_audio=None,
                 transform_image=None,
                 length=5,
                 **_):
        super(pseudo_dataset, self).__init__()
        self.data = file_list
        self.pseudo_label = pseduo_label
        self.label = label
        self.root = root
        self.sample_rate = sample_rate
        self.length = length
        self.transform_audio = transform_audio
        self.transform_image = transform_image
        self.length = length

    def __getitem__(self, index: int):
        fname = self.data[index]
        if self.transform_audio:
            fname_audio = self.root + fname + '.flac'
            if os.path.isfile(fname_audio):
                audio, sample_rate = librosa.load(fname_audio, sr=self.sample_rate)
                audio = np.pad(audio, (0, self.length * self.sample_rate - len(audio)))
                audio = (audio * 32768.0).astype(np.float32)[np.newaxis, :]
            else:
                audio = np.zeros((1, self.sample_rate * self.length))
            if self.transform_audio is not None and audio is not None:
                audio = self.transform_audio(audio)
            return audio, self.pseudo_label[index], self.label[index]
        else:
            fname_video = self.root + fname + '.mp4'
            vid = ffmpeg.probe(fname_video)
            center = float(vid['streams'][0]['duration']) / 2
            image, _, _ = tv.io.read_video(fname_video, start_pts=center, end_pts=center, pts_unit='sec')
            image = (image[0] / 255).permute(2, 0, 1)

            if self.transform_image is not None and image is not None:
                image = self.transform_image(image)
            return image, self.pseudo_label[index], self.label[index]
    def __len__(self) -> int:
        return len(self.data)
def train(train_data, test_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    model = resnet18()
    model.load_state_dict(torch.load('resnet18.pth'))
    model.fc = torch.nn.Linear(512, number_cls)
    model = model.to(device)

    transform_image = tv.transforms.Compose([
        tv.transforms.Resize(224, interpolation=tv.transforms.InterpolationMode.BICUBIC),
        tv.transforms.CenterCrop(224),
        tv.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    train_dataset = pseudo_dataset('../dataset/VggSound/', *train_data, transform_image=transform_image)
    test_dataset = pseudo_dataset('../dataset/VggSound/', *test_data, transform_image=transform_image)
    # len_train = int(len(dataset) * 0.8)
    # len_test = len(dataset) - len_train
    # train_dataset, test_dataset = td.random_split(dataset, [len_train, len_test],
    #                                               generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=16, shuffle=True,
                                               drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=16, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    for e in range(5):
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            data, pseudo_label, _ = batch
            data = data.to(device)
            predict = model(data)
            l = loss(predict, pseudo_label.to(device))
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
    embed = np.load('save_embedding.npz')
    audio = embed['audio']
    image = embed['image']
    text = embed['text']
    y = embed['y']
    name = embed['name']

    d = defaultdict(list)
    for i, x in enumerate(y.tolist()):
        d[x].append(i)
    grp = list(d.values())
    cls = list(d.keys())
    for i in range(20):
        number_cls = 10
        class_y, group_y = zip(*random.sample(list(zip(cls, grp)), number_cls))
        group_y = sum(group_y, [])
        def class_map(cls):
            return class_y.index(cls)
        image_select = image[group_y]; audio_select = audio[group_y]; name_select = name[group_y]
        y_select = np.array(list(map(class_map, y[group_y]))); text_select = text[list(class_y)]
        select, label = pseduo_label(image_select, audio_select, text_select, y_select, method='skewness')
        order = np.arange(len(group_y))
        np.random.shuffle(order)
        name_select = name_select[order]; label = label[order]; y_select = y_select[order];
        train_data = [name_select[:int(0.8 * len(group_y))], label[:int(0.8 * len(group_y))], y_select[:int(0.8 * len(group_y))]]
        test_data = [name_select[int(0.8 * len(group_y)):], label[int(0.8 * len(group_y)):], y_select[int(0.8 * len(group_y)):]]

        # train_data = [name_select[select], label[select], y_select[select]]
        # test_data = [name_select[~select], label[~select], y_select[~select]]
        # train(train_data, test_data)

