import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torchaudio
import torch.nn as nn
from data import NoisyCleanSet, read_transfer_function, IMUSPEECHSet
from A2net import A2net
import numpy as np
from result import baseline, vibvoice
from tqdm import tqdm
import argparse
import pickle
import os

def cal_loss(train_loader, test_loader, device, model, Loss, optimizer, scheduler):
    for x, noise, y in tqdm(train_loader):
            x, noise, y = x.to(device=device, dtype=torch.float), noise.to(device=device, dtype=torch.float), y.to(
                device=device, dtype=torch.float)
            predict1, predict2 = model(x, noise)
            loss1 = Loss(predict1, y)
            loss2 = Loss(predict2, y[:, :, :33, :])
            loss = loss1 + 0.05 * loss2
            #loss = loss1
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


def train(dataset, EPOCH, lr, BATCH_SIZE, Loss, device, model, save_all=False):

    length = len(dataset)
    test_size = min(int(0.2 * length), 2000)
    train_size = length - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], torch.Generator().manual_seed(0))
    train_loader = Data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.00001)
    loss_best = 1
    loss_curve = []
    ckpt_best = model.state_dict()
    for e in range(EPOCH):
        ckpt, loss1, loss2 = cal_loss(train_loader, test_loader, device, model, Loss, optimizer, scheduler)
        loss_curve.append(loss1)
        if loss1 < loss_best:
            ckpt_best = ckpt
            loss_best = loss1
            if save_all:
                torch.save(ckpt_best, 'pretrain/' + str(loss_curve[-1]) + '.pth')
    return ckpt_best, loss_curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-pre train, 1-main benchmark, 2-mirco benchmark')
    args = parser.parse_args()

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    Loss = nn.L1Loss()

    if args.mode == 0:
        BATCH_SIZE = 64
        lr = 0.005
        EPOCH = 30
        transfer_function, variance = read_transfer_function('../transfer_function')

        dataset = NoisyCleanSet(transfer_function, variance, 'speech100.json', 'background.json', alpha=(1, 0.1, 0.1, 0.1), ratio=1)
        model = nn.DataParallel(A2net()).to(device)
        #model = A2net().to(device)
        ckpt_best, loss_curve = train(dataset, EPOCH, lr, BATCH_SIZE, Loss, device, model, save_all=True)

        plt.plot(loss_curve)
        plt.savefig('loss.png')

    elif args.mode == 1:
        # train one by one
        file = open("clean_train_paras.pkl", "rb")
        norm_clean = pickle.load(file)
        file = open("noise_train_paras.pkl", "rb")
        norm_noise = pickle.load(file)

        source = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        candidate = ["he", "shi",  "hou", "1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai"]
        for target in candidate:
            support = [x for x in source if x != target]
            datasets = []
            for c in support:
                datasets.append(IMUSPEECHSet('noise_train_imuexp7.json', 'noise_train_gtexp7.json', 'noise_train_wavexp7.json', simulate=False, person=[c], minmax=norm_noise[c]))
            train_dataset = Data.ConcatDataset(datasets)
            user_dataset = IMUSPEECHSet('clean_train_imuexp7.json', 'clean_train_wavexp7.json', 'clean_train_wavexp7.json', person=target, minmax=norm_clean[target], ratio=1)
            model = nn.DataParallel(A2net()).to(device)
            # ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()
            #ckpt = torch.load("checkpoint/5min/he_70.05555555555556.pth")
            ckpt = torch.load('pretrain/0.0014942875871109583.pth')
            #ckpt = {'module.' + key: value for key, value in ckpt.items()}

            #
            # pth_file = 'pretrain/' + os.listdir('pretrain')[-1]
            # ckpt = torch.load(pth_file)

            model.load_state_dict(ckpt)
            ckpt, _ = train(train_dataset, 5, 0.0005, 16, Loss, device, model)
            model.load_state_dict(ckpt)
            ckpt, _ = train(user_dataset, 2, 0.0001, 4, Loss, device, model)
            model.load_state_dict(ckpt)
            PESQ, SNR, WER = vibvoice(model, target, 0)
            mean_PESQ = np.mean(PESQ)
            mean_WER = np.mean(WER)

            torch.save(ckpt, target + '_' + str(mean_WER) + '.pth')
            np.savez(target + '_' + str(mean_WER) + '.npz', PESQ=PESQ, SNR=SNR, WER=WER)
            print(target)
            print(mean_PESQ)
            print(mean_WER)
    elif args.mode == 2:
        # test on real noise
        ckpt = torch.load("checkpoint/5min/he_70.05555555555556.pth")
        for target in ['office', 'corridor', 'stair']:
            PESQ, WER = vibvoice(ckpt, target, 3)
            mean_PESQ = np.mean(PESQ, axis=0)
            mean_WER = np.mean(WER, axis=0)
            best = mean_WER[1]
            np.savez(target + '_' + str(best) + '.npz', PESQ=PESQ, WER=WER)
            print(target)
            print(mean_PESQ)
            print(mean_WER)

    elif args.mode == 3:
        # synthetic dataset
        file = open("clean_train_paras.pkl", "rb")
        norm_clean = pickle.load(file)
        candidate = ['he', 'hou']
        for target in candidate:
            train_dataset = IMUSPEECHSet('clean_train_imuexp7.json', 'clean_train_wavexp7.json', 'clean_train_wavexp7.json', person=[target],
                                         minmax=norm_clean[target])
            ckpt = torch.load("five_second/imu_extra/0.0014658941369513438_0.012233901943545789.pth")
            ckpt = train(train_dataset, 10, 0.0001, 4, Loss, device, ckpt)
            PESQ, WER = vibvoice(ckpt, target, 1)
            mean_PESQ = np.mean(PESQ, axis=0)
            mean_WER = np.mean(WER, axis=0)
            best = mean_WER[1]
            np.savez(target + '_' + str(best) + '_clean.npz', PESQ=PESQ, WER=WER)
            print(target)
            print(mean_PESQ)
            print(mean_WER)

            PESQ, WER = vibvoice(ckpt, target, 2)
            mean_PESQ = np.mean(PESQ, axis=0)
            mean_WER = np.mean(WER, axis=0)
            best = mean_WER[1]
            np.savez(target + '_' + str(best) + '_mobile.npz', PESQ=PESQ, WER=WER)
            print(target)
            print(mean_PESQ)
            print(mean_WER)
    else:
        # micro-benchmark for the three earphones
        file = open("noise_train_paras.pkl", "rb")
        norm_noise = pickle.load(file)
        candidate = ['airpod', 'freebud', 'galaxy']
        for target in candidate:
            train_dataset = IMUSPEECHSet('noise_train_imuexp7.json', 'noise_train_gtexp7.json', 'noise_train_wavexp7.json', simulate=False, person=[target], minmax=norm_noise[target])
            ckpt = torch.load("five_second/imu_extra/0.0014658941369513438_0.012233901943545789.pth")
            ckpt = train(train_dataset, 10, 0.0001, 4, Loss, device, ckpt)
            PESQ, WER = vibvoice(ckpt, target, 0)
            mean_PESQ = np.mean(PESQ, axis=0)
            mean_WER = np.mean(WER, axis=0)
            best = mean_WER[1]
            np.savez(target + '_' + str(best) + '_result.npz', PESQ=PESQ, WER=WER)
            print(target)
            print(mean_PESQ)
            print(mean_WER)






