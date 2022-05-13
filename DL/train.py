import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torchaudio
import torch.nn as nn
from data import NoisyCleanSet, read_transfer_function, IMUSPEECHSet
from A2net import A2net
import numpy as np
from result import baseline, vibvoice, offline_evaluation
from tqdm import tqdm
import argparse
import pickle
import os

def cal_loss(train_loader, test_loader, device, model, Loss, optimizer, scheduler):
    for x, noise, y in tqdm(train_loader):
            x, noise, y = x.to(device=device, dtype=torch.float), torch.abs(noise).to(device=device, dtype=torch.float), \
                          torch.abs(y).to(device=device, dtype=torch.float)
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
            x, noise, y = x.to(device=device, dtype=torch.float), torch.abs(noise).to(device=device, dtype=torch.float), \
                          torch.abs(y).to(device=device, dtype=torch.float)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
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
    pkl_folder = "pkl/stft/"
    #Loss = nn.MSELoss()
    if args.mode == 0:
        BATCH_SIZE = 64
        lr = 0.001
        EPOCH = 30
        dataset = NoisyCleanSet('json/speech100.json', 'json/all_noise.json', alpha=(1, 0.002, 0.002, 0.002), ratio=1)
        model = nn.DataParallel(A2net()).to(device)
        ckpt_best, loss_curve = train(dataset, EPOCH, lr, BATCH_SIZE, Loss, device, model, save_all=True)

        plt.plot(loss_curve)
        plt.savefig('loss.png')

    elif args.mode == 1:
        # train one by one
        file = open(pkl_folder + "clean_train_paras.pkl", "rb")
        norm_clean = pickle.load(file)
        file = open(pkl_folder + "noise_train_paras.pkl", "rb")
        norm_noise = pickle.load(file)

        source = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        candidate = [ "shi", "hou", "1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai"]
        for target in candidate:
            support = [x for x in source if x != target]
            datasets = []
            for c in support:
                datasets.append(IMUSPEECHSet('json/noise_train_imuexp7.json', 'json/noise_train_gtexp7.json', 'json/noise_train_wavexp7.json', simulate=False, person=[c], minmax=norm_noise[c]))
            train_dataset = Data.ConcatDataset(datasets)
            user_dataset = IMUSPEECHSet('json/clean_train_imuexp7.json', 'json/clean_train_wavexp7.json', 'json/clean_train_wavexp7.json', ratio=0.2, person=[target], minmax=norm_clean[target])
            model = nn.DataParallel(A2net()).to(device)
            ckpt = torch.load('pretrain/L1/0.0013439175563689787.pth')
            #ckpt = torch.load('pretrain/mel/0.0034707123340922408.pth')
            # for f in os.listdir('checkpoint/1min'):
            #     if f[-3:] == 'pth' and f[:len(target)] == target:
            #         pth_file = f
            # ckpt = torch.load('checkpoint/1min/' + pth_file)
            # ckpt = {'module.' + k: v for k, v in ckpt.items()}
            # print(pth_file)

            # model.load_state_dict(ckpt)
            # ckpt, _ = train(train_dataset, 5, 0.001, 32, Loss, device, model)
            # model.load_state_dict(ckpt)
            # ckpt, _ = train(user_dataset, 2, 0.0001, 4, Loss, device, model)
            # model.load_state_dict(ckpt)

            file = open(pkl_folder + "noise_paras.pkl", "rb")
            paras = pickle.load(file)
            dataset = IMUSPEECHSet('json/noise_imuexp7.json', 'json/noise_gtexp7.json', 'json/noise_wavexp7.json', person=[target], simulate=False, phase=True, minmax=paras[target])
            PESQ, SNR, WER = vibvoice(model, dataset)
            mean_WER = np.mean(WER)
            np.savez(target + '_' + str(mean_WER) + '.npz', PESQ=PESQ, SNR=SNR, WER=WER)
            print(target, mean_WER)

    elif args.mode == 2:
        # synthetic dataset
        file = open(pkl_folder + "clean_train_paras.pkl", "rb")
        norm_clean = pickle.load(file)
        source = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        datasets = []
        for target in source:
            datasets.append(IMUSPEECHSet('json/clean_train_imuexp7.json', 'json/clean_train_wavexp7.json', 'json/speech.json', person=[target], minmax=norm_clean[target]))
        train_dataset = Data.ConcatDataset(datasets)

        model = nn.DataParallel(A2net()).to(device)
        ckpt = torch.load('pretrain/L1/0.0013439175563689787.pth')
        model.load_state_dict(ckpt)
        ckpt, _ = train(train_dataset, 5, 0.001, 32, Loss, device, model)
        model.load_state_dict(ckpt)

        # test on real noise
        # file = open(pkl_folder + "noise_paras.pkl", "rb")
        # paras = pickle.load(file)
        # for target in source:
        #     dataset = IMUSPEECHSet('json/noise_imuexp7.json', 'json/noise_gtexp7.json', 'json/noise_wavexp7.json',
        #                            person=[target], simulate=False, phase=True, minmax=paras[target])
        #     PESQ, SNR, WER = vibvoice(model, dataset)
        #     mean_WER = np.mean(WER)
        #     np.savez(target + '_' + str(mean_WER) + '.npz', PESQ=PESQ, SNR=SNR, WER=WER)
        #     print(target, mean_WER)

        # change noise type
        # for target in source:
        #     datasets.append(IMUSPEECHSet('json/clean_train_imuexp7.json', 'json/clean_train_wavexp7.json', 'json/music.json',person=[target], minmax=norm_clean[target]))
        # train_dataset = Data.ConcatDataset(datasets)

        length = len(train_dataset)
        test_size = min(int(0.2 * length), 2000)
        train_size = length - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size], torch.Generator().manual_seed(0))

        # test on synthetic noise
        PESQ, SNR = offline_evaluation(model, test_dataset)
        mean_PESQ = np.mean(PESQ, axis=0)
        mean_SNR = np.mean(SNR, axis=0)
        np.savez(str(mean_PESQ) + '.npz', PESQ=PESQ, SNR=SNR)
        print(mean_PESQ, mean_SNR)

    elif args.mode == 3:
        # test on audio-only, the acceleration is generated by transfer function
        file = open(pkl_folder + "clean_train_paras.pkl", "rb")
        norm_clean = pickle.load(file)
        source = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        datasets = []
        for target in source:
            datasets.append(IMUSPEECHSet('json/clean_train_imuexp7.json', 'json/clean_train_wavexp7.json', 'json/speech.json', person=[target], minmax=norm_clean[target]))
        dataset = Data.ConcatDataset(datasets)
        model = nn.DataParallel(A2net()).to(device)
        ckpt = torch.load('pretrain/L1/0.0013439175563689787.pth')
        model.load_state_dict(ckpt)
        ckpt_best, loss_curve = train(dataset, 5, 0.001, 32, Loss, device, model)
        model.load_state_dict(ckpt_best)

        file = open(pkl_folder + "clean_paras.pkl", "rb")
        norm_clean = pickle.load(file)
        for target in ['he', 'hou']:
            dataset = NoisyCleanSet('json/clean_wavexp7.json', 'json/speech.json', phase=True, alpha=[1] + norm_clean[target])
            PESQ, SNR, WER = vibvoice(model, dataset)
            mean_PESQ = np.mean(PESQ)
            mean_WER = np.mean(WER)
            np.savez(target + '_' + str(mean_WER) + '_augmented_IMU.npz', PESQ=PESQ, SNR=SNR, WER=WER)
            print(target, mean_PESQ, mean_WER)

    elif args.mode == 4:
        # field test
        ckpt = torch.load("checkpoint/5min/he_70.05555555555556.pth")
        for target in ['office', 'corridor', 'stair']:
            PESQ, WER = vibvoice(ckpt, target, 4)
            mean_PESQ = np.mean(PESQ)
            mean_WER = np.mean(WER)
            np.savez(target + '_' + str(mean_WER) + '.npz', PESQ=PESQ, WER=WER)
            print(target, mean_PESQ, mean_WER)

    else:
        # micro-benchmark for the three earphones
        file = open(pkl_folder + "noise_train_paras.pkl", "rb")
        norm_noise = pickle.load(file)
        candidate = ['airpod', 'freebud', 'galaxy']
        model = nn.DataParallel(A2net()).to(device)
        ckpt = torch.load('checkpoint/he_66.58333333333333.pth')
        model.load_state_dict(ckpt)
        for target in candidate:
            file = open(pkl_folder + "noise_paras.pkl", "rb")
            paras = pickle.load(file)
            dataset = IMUSPEECHSet('json/noise_imuexp7.json', 'json/noise_gtexp7.json', 'json/noise_wavexp7.json', person=[target], simulate=False, phase=True)
            PESQ, SNR, WER = vibvoice(model, dataset)
            mean_WER = np.mean(WER)
            np.savez(target + '_' + str(mean_WER) + '.npz', PESQ=PESQ, WER=WER)
            print(target)






