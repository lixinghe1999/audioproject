from torchvision.models import resnet18
import numpy as np
from pseudo import pseduo_label
import torch.utils.data as td
import torchvision as tv
import ffmpeg
import os
import librosa
import torch
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
            fname_audio = self.root + fname[:-3] + 'wav'
            if os.path.isfile(fname_audio):
                audio, sample_rate = librosa.load(fname_audio, sr=self.sample_rate, offset=center - self.length/2, duration=self.length)
                audio = np.pad(audio, (0, self.length * self.sample_rate - len(audio)))
                audio = (audio * 32768.0).astype(np.float32)[np.newaxis, :]
            else:
                audio = np.zeros((1, self.sample_rate * self.length))
            if self.transform_audio is not None and audio is not None:
                audio = self.transform_audio(audio)
            return audio, self.pseudo_label[index], self.label[index]
        else:
            fname_video = self.root + fname
            vid = ffmpeg.probe(fname_video)
            center = float(vid['streams'][0]['duration']) / 2
            image, _, _ = tv.io.read_video(fname_video, start_pts=center, end_pts=center, pts_unit='sec')
            image = (image[0] / 255).permute(2, 0, 1)

            if self.transform_image is not None and image is not None:
                image = self.transform_image(image)
            return image, self.pseudo_label[index], self.label[index]
    def __len__(self) -> int:
        return len(self.data)
def train(train_loader, test_loader, optimizer, scheduler):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        data, pseudo_label, label = batch
        data = data.to(device)
        predict = torch.argmax(model(data), dim=-1)

        print(predict, pseudo_label)
        loss = torch.nn.CrossEntropyLoss(predict, pseudo_label)
        loss.backward()
        optimizer.step()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            data, pseudo_label, label = batch
            data = data.to(device)
            predict = model(data)
            acc = (predict == label).sum()
            print(acc)


if __name__ == "__main__":
    embed = np.load('save_embedding.npz')
    audio = embed['audio']
    image = embed['image']
    text = embed['text']
    y = embed['y']
    name = embed['name']
    select, label = pseduo_label(image, text, y, method='skewness')
    name_select = name[select]; label_pseudo = label[select]; label_gt=y[select]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    model = resnet18()
    model.load_state_dict(torch.load('resnet18.pth'))
    model.fc = torch.nn.Linear(512, 101)
    model = model.to(device)

    transform_image = tv.transforms.Compose([
        tv.transforms.Resize(224, interpolation=tv.transforms.InterpolationMode.BICUBIC),
        tv.transforms.CenterCrop(224),
        tv.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    dataset = pseudo_dataset('../dataset/UCF101/', name_select, label_pseudo, label_gt, transform_image=transform_image)
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = td.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=16, shuffle=True,
                                         drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=16, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    train(train_loader, test_loader, optimizer, scheduler)
