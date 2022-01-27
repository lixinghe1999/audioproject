import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torchaudio
import torch.nn as nn
from data import NoisyCleanSet, read_transfer_function, IMUSPEECHSet
from unet import UNet
from A2net import A2net
from A2netU import A2netU
from A2netcomplex import A2net_m, A2net_p
import numpy as np
from tqdm import tqdm
import argparse

# exp6
# for mel spectrogram
# max for IMU 0.28
# max for SPEECH 0.73

# for spectrogram
# max for IMU 0.0132
# max for SPEECH 0.0037

# exp7
# for spectrogram - he
# max for IMU 0.0143
# max for SPEECH 0.045

# for spectrogram - hou
# max for IMU 0.0165
# max for SPEECH 0.123

# for Librispeech
# narrow band
# 0.0637 ratio = 17.2, 0.07
# wide band
# 0.0925 ratio = 17.2, 0.0116
def train_magnitude(train_loader, test_loader, device, model, Loss, optimizer, scheduler):
    for x, noise, y in tqdm(train_loader):
            x, noise, y = x.to(device=device, dtype=torch.float), noise.to(device=device, dtype=torch.float), y.to(
                device=device, dtype=torch.float)
            predict1 = model(x, noise)
            loss1 = Loss(predict1, y)
            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        Loss_all = []
        for x, noise, y in test_loader:
            x, noise, y = x.to(device=device, dtype=torch.float), noise.to(device=device, dtype=torch.float), y.to(
                device=device, dtype=torch.float)
            predict1 = model(x, noise)
            loss1 = Loss(predict1, y)
            Loss_all.append(loss1.item())
        val_loss = np.mean(Loss_all, axis=0)
    scheduler.step()
    return model.state_dict(), val_loss

def train_extra(train_loader, test_loader, device, model, Loss, optimizer, scheduler):
    for x, noise, y in tqdm(train_loader):
            x, noise, y = x.to(device=device, dtype=torch.float), noise.to(device=device, dtype=torch.float), y.to(
                device=device, dtype=torch.float)
            predict1, predict2 = model(x, noise)
            loss1 = Loss(predict1, y)
            loss2 = Loss(predict2, y[:, :, :33, :])
            loss = loss1 + 0.05 * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        Loss_all = []
        for x, noise, y in test_loader:
            x, noise, y = x.to(device=device, dtype=torch.float), noise.to(device=device, dtype=torch.float), y.to(
                device=device, dtype=torch.float)
            predict1, predict2 = model(x, noise)
            loss1 = Loss(predict1, y)
            loss2 = Loss(predict2, y[:, :, :33, :])
            Loss_all.append([loss1.item(), loss2.item()])
        val_loss = np.mean(Loss_all, axis=0)
    scheduler.step()
    return model.state_dict(), val_loss[0], val_loss[1]

def train_audio(train_loader, test_loader, device, model, Loss, optimizer, scheduler):
    for x, noise, y in tqdm(train_loader):
            noise, y = noise.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
            predict1 = model(x, noise)
            loss1 = Loss(predict1, y)
            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        Loss_all = []
        for x, noise, y in test_loader:
            noise, y = noise.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
            predict1 = model(x, noise)
            loss1 = Loss(predict1, y)
            Loss_all.append(loss1.item(),)
        val_loss = np.mean(Loss_all)
    scheduler.step()
    return model.state_dict(), val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-pre train, 1-fine-tune')
    args = parser.parse_args()
    EPOCH = 10
    BATCH_SIZE = 4
    lr = 0.0001
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    Loss = nn.L1Loss()

    #model = UNet(1, 1).to(device)
    #model.load_state_dict(torch.load("checkpoints/pretrain_0.011717716008242146.pth"))

    model1 = A2net_m(extra_supervision=True).to(device)
    model2 = A2net_p().to(device)
    #model1.load_state_dict(torch.load("vanilla/0.001456301872167387.pth"))
    #model1.load_state_dict(torch.load("audio/0.002334692117680485.pth"))
    model1.load_state_dict(torch.load("imu_extra/0.0014658941369513438_0.012233901943545789.pth"))

    if args.mode == 0:
        loss_curve = []
        transfer_function, variance = read_transfer_function('../transfer_function')
        train_dataset = NoisyCleanSet(transfer_function, variance, 'speech100.json', 'background.json', alpha=(28, 0.04, 0.095, 0.095))
        test_dataset = NoisyCleanSet(transfer_function, variance, 'devclean.json', 'background.json', alpha=(28, 0.04, 0.095, 0.095))
        train_loader = Data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
        optimizer = torch.optim.AdamW(model1.parameters(), lr=lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.00001)
        loss_best = 1
        for e in range(EPOCH):
            ckpt, loss1, loss2 = train_extra(train_loader, test_loader, device, model1, Loss, optimizer, scheduler)
            loss_curve.append(loss1)
            if loss1 < loss_best:
                ckpt_best = ckpt
                loss_best = loss1
                torch.save(ckpt_best, str(loss_best) + '_' + str(loss2) + '.pth')
        plt.plot(loss_curve)
        plt.savefig('loss.png')

        #model1.load_state_dict(torch.load("36_0.001661749362685908.pth"))
        # for p in model1.parameters():
        #     p.requires_grad = False
        # train_dataset = NoisyCleanSet(transfer_function, variance, 'speech100.json', 'reverse_speech100.json', phase=True, alpha=(28.78, 0.029, 0.095))
        # test_dataset = NoisyCleanSet(transfer_function, variance, 'devclean.json', 'reverse_devclean.json', phase=True, alpha=(28.78, 0.029, 0.095))
        # train_loader = Data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        # test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
        # optimizer = torch.optim.AdamW(model2.parameters(), lr=lr, weight_decay=0.05)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.00001)
        # loss_best = 1
        # for e in range(EPOCH):
        #     ckpt, loss = train_phase(train_loader, test_loader, device, model1, model2, Loss, optimizer, scheduler)
        #     if loss < loss_best:
        #         ckpt_best = ckpt
        #         loss_best = loss
        #         torch.save(ckpt_best, str(loss_best) + '.pth')





    elif args.mode==1:
        optimizer = torch.optim.AdamW(model1.parameters(), lr=lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.00001)

        #for c in ["liang", "wu", "he", "hou", "zhao", "shi"]:
        norm = {"he": (0.0186, 0.107, 0.107), "hou": (0.025, 0.11, 0.11), "shi": (0.0076, 0.087, 0.087), "shuai": (0.1, 0.089, 0.089)}
        for c in ["shi", "shuai"]:
            train_dataset = IMUSPEECHSet('train_imuexp7.json', 'train_wavexp7.json', 'train_wavexp7.json', person=[c], minmax=norm[c])
            length = len(train_dataset)
            train_size, validate_size = int(0.8 * length), int(0.2 * length)
            train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, validate_size], torch.Generator().manual_seed(0))
            train_loader = Data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
            test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
            loss_best = 1
            for e in range(EPOCH):
                ckpt, loss1, loss2 = train_extra(train_loader, test_loader, device, model1, Loss, optimizer, scheduler)
                if loss1 < loss_best:
                    ckpt_best = ckpt
                    loss_best = loss1
            torch.save(ckpt_best, c + '_' + str(loss_best) + '.pth')




