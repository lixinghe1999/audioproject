import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from dataset import NoisyCleanSet
from evaluation import SI_SDR
import os
'''
this script describe the deep learning mapping proposed by SEANet(google)
'''

class ResUnit(nn.Module):
    def __init__(self, input, output, dilation):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv1d(input, output, kernel_size=3, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(output, output, kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_channel, output_channel, S):
        super(EncoderBlock, self).__init__()
        self.re1 = ResUnit(input_channel, output_channel // 2, dilation=1)
        self.re2 = ResUnit(output_channel // 2, output_channel // 2, dilation=3)
        self.re3 = ResUnit(output_channel // 2, output_channel // 2, dilation=9)
        self.conv = nn.Conv1d(output_channel // 2, output_channel, kernel_size=2*S, stride=S, padding=S//2)
    def forward(self, x):
        x = self.re1(x)
        x = self.re2(x)
        x = self.re3(x)
        x = self.conv(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, input_channel, output_channel, S):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose1d(input_channel, output_channel, kernel_size=2 * S, stride=S, padding=S//2)
        self.re1 = ResUnit(output_channel, output_channel, dilation=1)
        self.re2 = ResUnit(output_channel, output_channel, dilation=3)
        self.re3 = ResUnit(output_channel, output_channel, dilation=9)
    def forward(self, x):
        x = self.conv(x)
        x = self.re1(x)
        x = self.re2(x)
        x = self.re3(x)
        return x

class SEANet_mapping(nn.Module):
    def __init__(self):
        super(SEANet_mapping, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.E1 = EncoderBlock(32, 64, 2)
        self.E2 = EncoderBlock(64, 128, 2)
        self.E3 = EncoderBlock(128, 256, 8)
        self.E4 = EncoderBlock(256, 512, 10)

        self.conv2 = nn.Conv1d(512, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(128, 512, kernel_size=7, padding=3)

        self.D1 = DecoderBlock(512, 128, 8)
        self.D2 = DecoderBlock(128, 64, 2)
        self.D3 = DecoderBlock(64, 32, 2)
        self.conv4 = nn.Conv1d(32, 3, kernel_size=7, padding=3)
        self.m = nn.AvgPool1d(10, stride=10)

    def forward(self, audio):
        # down-sample
        #acc = torch.nn.functional.interpolate(acc, scale_factor=10)
        x1 = audio
        x2 = self.conv1(x1)
        x3 = self.E1(x2)
        x4 = self.E2(x3)
        x5 = self.E3(x4)
        x6 = self.E4(x5)
        x = self.conv3(self.conv2(x6)) + x6
        # up-sample, may need padding if the duration is not * 256
        x = self.D1(x) + self.m(x4)
        x = self.D2(x) + self.m(x3)
        x = self.D3(x) + self.m(x2)
        x = self.conv4(x) + self.m(x1)
        return x
def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def train(dataset, EPOCH, lr, BATCH_SIZE, model, save_all=False):
    # deep learning-based mapping: from audio to acc
    if isinstance(dataset, list):
        # with pre-defined train/ test
        train_dataset, test_dataset = dataset
    else:
        # without pre-defined train/ test
        length = len(dataset)
        test_size = min(int(0.1 * length), 2000)
        train_size = length - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                                   pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    loss_best = 1
    loss_curve = []
    ckpt_best = model.state_dict()
    for e in range(EPOCH):
        for i, (acc, noise, clean) in enumerate(tqdm(train_loader)):
            clean = clean.to(device=device, dtype=torch.float)
            acc = acc.to(device=device, dtype=torch.float)
            optimizer.zero_grad()
            predict = model(clean)
            loss = nn.functional.mse_loss(predict, acc)
            loss.backward()
            optimizer.step()
        scheduler.step()
        Loss_list = []
        with torch.no_grad():
            for acc, noise, clean in test_loader:
                clean = clean.to(device=device, dtype=torch.float)
                acc = acc.to(device=device, dtype=torch.float)
                predict = model(clean)
                loss = nn.functional.l1_loss(predict, acc).item()
                Loss_list.append(loss)
        mean_lost = np.mean(Loss_list)
        loss_curve.append(mean_lost)
        print(mean_lost)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            if save_all:
                torch.save(ckpt_best, 'pretrain/' + str(loss_curve[-1]) + '.pth')
    return ckpt_best, loss_curve

def test(dataset, BATCH_SIZE, model):
    # deep learning-based mapping: from audio to acc
    if isinstance(dataset, list):
        # with pre-defined train/ test
        train_dataset, test_dataset = dataset
    else:
        # without pre-defined train/ test
        length = len(dataset)
        test_size = min(int(0.1 * length), 2000)
        train_size = length - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    sdr_list = []
    with torch.no_grad():
        for acc, noise, clean in test_loader:
            clean = clean.to(device=device, dtype=torch.float)
            acc = acc.to(device=device, dtype=torch.float)
            predict = model(clean)
            si_sdr = SI_SDR(acc.cpu().numpy(), predict.cpu().numpy())
            print(si_sdr)
            sdr_list.append(si_sdr)
    return
if __name__ == '__main__':
    model = SEANet_mapping()
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # This script is for model pre-training on LibriSpeech
    BATCH_SIZE = 16
    lr = 0.0001
    EPOCH = 30
    people = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
    dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'], time_domain=True,
                            person=people, simulation=True)
    model = model.to(device)

    # ckpt_best, loss_curve = train(dataset, EPOCH, lr, BATCH_SIZE, model, save_all=True)

    # For the testing: Si-SDR
    ckpt_dir = 'pretrain/deep_augmentation'
    ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[0]
    print("load checkpoint: {}".format(ckpt_name))
    ckpt = torch.load(ckpt_name)
    model.load_state_dict(ckpt)
    test(dataset, BATCH_SIZE, model)