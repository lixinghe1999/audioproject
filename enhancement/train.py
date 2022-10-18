import os
import time

import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
import torch.utils.data as Data
import torchaudio
import torch.nn as nn
from dataset import NoisyCleanSet
from fullsubnet import FullSubNet
from A2net import A2net
from causal_A2net import Causal_A2net
import numpy as np
import scipy.signal as signal
from result import subjective_evaluation, objective_evaluation
from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM
from tqdm import tqdm
import argparse
from evaluation import wer, snr, lsd, SI_SDR
from pesq import pesq_batch, pesq


seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32
rate_mic = 16000
rate_imu = 1600


freq_bin_high = 8 * int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1

def sample_evaluation(model, x, noise, y, audio_only=False, complex=False):
    x = x.to(device=device, dtype=torch.float)
    magnitude = torch.abs(noise).to(device=device, dtype=torch.float)
    phase = torch.angle(noise).to(device=device, dtype=torch.float)
    noise_real = noise.real.to(device=device, dtype=torch.float)
    noise_imag = noise.imag.to(device=device, dtype=torch.float)
    y = y.to(device=device).squeeze(1)
    if audio_only:
        predict1 = model(magnitude)
    else:
        predict1 = model(x, magnitude)
    # predict1 = magnitude
    # either predict the spectrogram, or predict the CIRM
    if complex:
        cRM = decompress_cIRM(predict1.permute(0, 2, 3, 1))
        enhanced_real = cRM[..., 0] * noise_real.squeeze(1) - cRM[..., 1] * noise_imag.squeeze(1)
        enhanced_imag = cRM[..., 1] * noise_real.squeeze(1) + cRM[..., 0] * noise_imag.squeeze(1)
        predict1 = torch.complex(enhanced_real, enhanced_imag)
    else:
        predict1 = torch.exp(1j * phase[:, :, :freq_bin_high, :]) * predict1
        predict1 = predict1.squeeze(1)

    predict = predict1.cpu().numpy()
    predict = np.pad(predict, ((0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
    _, predict = signal.istft(predict, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

    y = y.cpu().numpy()
    y = np.pad(y, ((0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
    _, y = signal.istft(y, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)
    return np.stack([np.array(pesq_batch(16000, y, predict, 'wb', on_error=1)), SI_SDR(y, predict), lsd(y, predict)], axis=1)

def sample(model, x, noise, y, audio_only=False):
    cIRM = build_complex_ideal_ratio_mask(noise.real, noise.imag, y.real, y.imag)  # [B, 2, F, T]
    cIRM = cIRM.to(device=device, dtype=torch.float)
    x = x.to(device=device, dtype=torch.float)
    noise = noise.abs().to(device=device, dtype=torch.float)
    y = y.abs().to(device=device, dtype=torch.float)

    if audio_only:
        predict1 = model(noise)
        loss = Loss(predict1, cIRM)
    else:
        predict1 = model(x, noise)
        loss1 = Loss(predict1, y)
        #loss2 = Loss(predict2, y[:, :, :32, :])
        loss = loss1

    return loss

def train(dataset, EPOCH, lr, BATCH_SIZE, model, save_all=False, audio_only=False, complex=False):
    if isinstance(dataset, list):
        # with pre-defined train/ test
        train_dataset, test_dataset = dataset
    else:
        # without pre-defined train/ test
        length = len(dataset)
        test_size = min(int(0.1 * length), 2000)
        train_size = length - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = Data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    #optimizer = torch.optim.Adam(params= filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    loss_best = 1
    loss_curve = []
    ckpt_best = model.state_dict()
    for e in range(EPOCH):
        Loss_list = []
        for i, (x, noise, y) in enumerate(tqdm(train_loader)):
            loss = sample(model, x, noise, y, audio_only=audio_only)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss_list.append(loss.item())
        mean_lost = np.mean(Loss_list)
        loss_curve.append(mean_lost)
        Metric = []
        with torch.no_grad():
            for x, noise, y in tqdm(test_loader):
                metric = sample_evaluation(model, x, noise, y, audio_only=audio_only, complex=complex)
                Metric.append(metric)
        scheduler.step()
        avg_metric = np.mean(np.concatenate(Metric, axis=0), axis=0)
        print(avg_metric)

        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            metric_best = avg_metric
            if save_all:
                torch.save(ckpt_best, 'pretrain/' + str(loss_curve[-1]) + '.pth')
    torch.save(ckpt_best, 'pretrain/' + str(metric_best) + '.pth')
    return ckpt_best, loss_curve, metric_best

def inference(dataset, BATCH_SIZE, model, audio_only=False, complex=False):
    # length = len(dataset)
    # test_size = min(int(0.1 * length), 2000)
    # train_size = length - test_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    test_dataset = dataset
    test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    Metric = []
    with torch.no_grad():
        for x, noise, y in test_loader:
            metric = sample_evaluation(model, x, noise, y, audio_only=audio_only, complex=complex)
            Metric.append(metric)
    Metric = np.concatenate(Metric, axis=0)
    return Metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-pre train, 1-main benchmark, 2-mirco benchmark')
    args = parser.parse_args()
    audio_only = False
    complex = False
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    Loss = nn.MSELoss()
    torch.cuda.set_device(1)
    if args.mode == 0:
        # This script is for model pre-training on LibriSpeech
        BATCH_SIZE = 64
        lr = 0.01
        EPOCH = 30
        dataset = NoisyCleanSet(['json/train.json', 'json/all_noise.json'], simulation=True, ratio=0.1)

        #model = A2net(inference=False).to(device)
        #model = FullSubNet(num_freqs=256).to(device)
        #model = Sequence_A2net().to(device)
        model = Causal_A2net(inference=True).to(device)
        ckpt_best, loss_curve, metric_best = train(dataset, EPOCH, lr, BATCH_SIZE, model,
                                                   save_all=True, audio_only=audio_only, complex=complex)
        plt.plot(loss_curve)
        plt.savefig('loss.png')

    elif args.mode == 1:
        # This script is for model fine-tune on self-collected dataset, by default-with all noises
        BATCH_SIZE = 16
        lr = 0.001
        EPOCH = 10

        ckpt_dir = 'pretrain/fullsubnet'
        ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[0]
        ckpt = torch.load(ckpt_name)

        #model = A2net(inference=False).to(device)
        model = FullSubNet(num_freqs=264).to(device)

        people = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        train_dataset1 = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'], person=people, simulation=True, ratio=0.8)
        test_dataset1 = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'], person=people, simulation=True, ratio=-0.2)

        # extra dataset for other positions
        positions = ['glasses', 'vr-up', 'vr-down', 'headphone-inside', 'headphone-outside', 'cheek', 'temple', 'back', 'nose']
        train_dataset2 = NoisyCleanSet(['json/position_gt.json', 'json/all_noise.json', 'json/position_imu.json'], person=positions, simulation=True, ratio=0.8)
        test_dataset2 = NoisyCleanSet(['json/position_gt.json', 'json/all_noise.json', 'json/position_imu.json'],person=positions, simulation=True, ratio=-0.2)

        train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
        test_dataset = torch.utils.data.ConcatDataset([test_dataset1, test_dataset2])

        model.load_state_dict(ckpt)
        ckpt, loss_curve, metric_best = train([train_dataset, test_dataset], EPOCH, lr, BATCH_SIZE, model, audio_only=audio_only, complex=complex)

        # Optional Micro-benchmark
        model.load_state_dict(ckpt)
        people = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        for p in people:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'], person=[p], simulation=True, ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model, audio_only=audio_only, complex=complex)
            avg_metric = np.mean(Metric, axis=0)
            print(p, avg_metric)

        for noise in ['background.json', 'dev.json', 'music.json']:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/' + noise,  'json/train_imu.json'], person=people, simulation=True, ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model,  audio_only=audio_only, complex=complex)
            avg_metric = np.mean(Metric, axis=0)
            print(noise, avg_metric)

        for level in [11, 6, 1]:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json',  'json/train_imu.json'], person=people,
                                    simulation=True, snr=[level - 1, level + 1], ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model,  audio_only=audio_only, complex=complex)
            avg_metric = np.mean(Metric, axis=0)
            print(level, avg_metric)

        # Micro-benchmark for different positions
        positions = ['glasses', 'vr-up', 'vr-down', 'headphone-inside', 'headphone-outside', 'cheek', 'temple', 'back', 'nose']
        for p in positions:
            dataset = NoisyCleanSet(['json/position_gt.json', 'json/all_noise.json', 'json/position_imu.json'], person=[p], simulation=True, ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model,  audio_only=audio_only, complex=complex)
            avg_metric = np.mean(Metric, axis=0)
            print(p, avg_metric)

    elif args.mode == 2:
        # micro-benchmark per-user, length of data
        BATCH_SIZE = 32
        lr = 0.001
        EPOCH = 3

        ckpt_dir = 'pretrain/vibvoice_new'
        ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[0]
        ckpt_start = torch.load(ckpt_name)

        model = nn.DataParallel(A2net()).to(device)
        #model = nn.DataParallel(Model(num_freqs=264).to(device), device_ids=[0, 1])

        # synthetic dataset
        result = []
        people = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        for p in people:
            model.load_state_dict(ckpt_start)
            p_except = [i for i in people if i != p]
            train_dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'], person=p_except, simulation=True)
            test_dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'], person=[p], simulation=True)

            # involve part of the target user data
            # length = len(test_dataset)
            # train_size = int(0.1 * length)
            # test_size = length - train_size
            # train_dataset_target, test_dataset = torch.utils.data.random_split(test_dataset, [train_size, test_size])
            # train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_target])

            _, _, avg_metric = train([train_dataset, test_dataset], EPOCH, lr, BATCH_SIZE, model, audio_only=False, complex=False)
            result.append(avg_metric)
        print('average performance for all users: ', np.mean(result, axis=0))
    elif args.mode == 3:
        BATCH_SIZE = 32
        lr = 0.001
        EPOCH = 10

        ckpt_dir = 'pretrain/vibvoice_new'
        ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[0]
        ckpt = torch.load(ckpt_name)

        model = nn.DataParallel(A2net()).to(device)
        # model = nn.DataParallel(Model(num_freqs=264).to(device), device_ids=[0, 1])

        people = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'], person=people, simulation=True)

        model.load_state_dict(ckpt)
        ckpt, loss_curve, metric_best = train(dataset, EPOCH, lr, BATCH_SIZE, model, audio_only=False, complex=False)

        people = ['he']


        test_dataset1 = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'], person=people,
                                     simulation=True)
        test_dataset2 = NoisyCleanSet(['json/test_gt.json', 'json/all_noise.json', 'json/test_imu.json'], person=people,
                                      simulation=True)
        test_dataset3 = NoisyCleanSet(['json/mask_gt.json', 'json/all_noise.json', 'json/mask_imu.json'], person=people,
                                      simulation=True)
        # model.load_state_dict(torch.load('pretrain/[ 2.42972495 15.36821378  4.22121219].pth'))
        model.load_state_dict(ckpt)
        avg_metric1 = inference(test_dataset1, BATCH_SIZE, model, audio_only=False, complex=False)
        print('first time performance', avg_metric1)
        avg_metric2 = inference(test_dataset2, BATCH_SIZE, model, audio_only=False, complex=False)
        print('second time performance', avg_metric2)
        avg_metric3 = inference(test_dataset3, BATCH_SIZE, model, audio_only=False, complex=False)
        print('mask-on performance', avg_metric3)

    # elif args.mode == 2:
    #     # train one by one
    #     file = open(pkl_folder + "clean_train_paras.pkl", "rb")
    #     norm_clean = pickle.load(file)
    #     file = open(pkl_folder + "noise_train_paras.pkl", "rb")
    #     norm_noise = pickle.load(file)
    #
    #     source = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
    #     #source = ["hou", "he"]
    #     candidate = ["he", "shi", "hou", "1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai"]
    #     #candidate = ['hou']
    #     for target in candidate:
    #         source = [x for x in source if x != target]
    #         datasets = []
    #         for c in source:
    #             datasets.append(IMUSPEECHSet('json/noise_train_imu.json', 'json/noise_train_gt.json', 'json/noise_train_wav.json', simulate=False, person=[c], norm=norm_noise[c]))
    #         train_dataset = Data.ConcatDataset(datasets)
    #         user_dataset = IMUSPEECHSet('json/train_imu.json', 'json/train_wav.json', 'json/train_wav.json', ratio=1, person=[target], norm=norm_clean[target])
    #         model = nn.DataParallel(A2net()).to(device)
    #         #ckpt = torch.load('pretrain/L1/0.0013439175563689787.pth')
    #         #ckpt = torch.load('pretrain/mel/0.0034707123340922408.pth')
    #         ckpt = torch.load('pretrain/0.06400973408017308.pth')
    #         # for f in os.listdir('checkpoint/5min'):
    #         #     if f[-3:] == 'pth' and f[:len(target)] == target:
    #         #         pth_file = f
    #         # ckpt = torch.load('checkpoint/5min/' + pth_file)
    #         # ckpt = {'module.' + k: v for k, v in ckpt.items()}
    #         # print(pth_file)
    #         model.load_state_dict(ckpt)
    #
    #         # ckpt, _ = train(train_dataset, 5, 0.001, 32, Loss, device, model)
    #         # model.load_state_dict(ckpt)
    #         # ckpt, _ = train(user_dataset, 2, 0.0001, 4, Loss, device, model)
    #         # model.load_state_dict(ckpt)
    #
    #         file = open(pkl_folder + "noise_paras.pkl", "rb")
    #         paras = pickle.load(file)
    #         dataset = IMUSPEECHSet('json/noise_imu.json', 'json/noise_gt.json', 'json/noise_wav.json', person=[target], simulate=False, phase=True, norm=paras[target])
    #         WER = subjective_evaluation(model, dataset)
    #         mean_WER = np.mean(WER)
    #         np.savez(target + '_' + str(mean_WER) + '.npz', PESQ=PESQ, SNR=SNR, WER=WER)
    #         print(target, mean_WER)
    #
    #
    #
    # elif args.mode == 3:
    #     # test on audio-only, the acceleration is generated by transfer function
    #     file = open(pkl_folder + "clean_train_paras.pkl", "rb")
    #     norm_clean = pickle.load(file)
    #     source = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
    #     datasets = []
    #     for target in source:
    #         datasets.append(IMUSPEECHSet('json/clean_train_imuexp7.json', 'json/clean_train_wavexp7.json', 'json/speech.json', person=[target], minmax=norm_clean[target]))
    #     dataset = Data.ConcatDataset(datasets)
    #     model = nn.DataParallel(A2net()).to(device)
    #     ckpt = torch.load('pretrain/L1/0.0013439175563689787.pth')
    #     model.load_state_dict(ckpt)
    #     ckpt_best, loss_curve = train(dataset, 5, 0.001, 32, Loss, device, model)
    #     model.load_state_dict(ckpt_best)
    #
    #     file = open(pkl_folder + "clean_paras.pkl", "rb")
    #     norm_clean = pickle.load(file)
    #     for target in ['he', 'hou']:
    #         dataset = NoisyCleanSet('json/clean_wavexp7.json', 'json/speech.json', phase=True, alpha=[1] + norm_clean[target])
    #         PESQ, SNR, WER = vibvoice(model, dataset)
    #         mean_PESQ = np.mean(PESQ)
    #         mean_WER = np.mean(WER)
    #         np.savez(target + '_' + str(mean_WER) + '_augmented_IMU.npz', PESQ=PESQ, SNR=SNR, WER=WER)
    #         print(target, mean_PESQ, mean_WER)
    #
    # elif args.mode == 4:
    #     # field test
    #     ckpt = torch.load("checkpoint/5min/he_70.05555555555556.pth")
    #     for target in ['office', 'corridor', 'stair']:
    #         PESQ, WER = vibvoice(ckpt, target, 4)
    #         mean_PESQ = np.mean(PESQ)
    #         mean_WER = np.mean(WER)
    #         np.savez(target + '_' + str(mean_WER) + '.npz', PESQ=PESQ, WER=WER)
    #         print(target, mean_PESQ, mean_WER)
    #
    # else:
    #     # micro-benchmark for the three earphones
    #     file = open(pkl_folder + "noise_train_paras.pkl", "rb")
    #     norm_noise = pickle.load(file)
    #     candidate = ['airpod', 'freebud', 'galaxy']
    #     model = nn.DataParallel(A2net()).to(device)
    #     ckpt = torch.load('checkpoint/he_66.58333333333333.pth')
    #     model.load_state_dict(ckpt)
    #     for target in candidate:
    #         file = open(pkl_folder + "noise_paras.pkl", "rb")
    #         paras = pickle.load(file)
    #         dataset = IMUSPEECHSet('json/noise_imuexp7.json', 'json/noise_gtexp7.json', 'json/noise_wavexp7.json', person=[target], simulate=False, phase=True)
    #         PESQ, SNR, WER = vibvoice(model, dataset)
    #         mean_WER = np.mean(WER)
    #         np.savez(target + '_' + str(mean_WER) + '.npz', PESQ=PESQ, WER=WER)
    #         print(target)
    #
    #




