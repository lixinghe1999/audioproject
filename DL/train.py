import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torchaudio
import torch.nn as nn
from dataset import NoisyCleanSet
from fullsubnet import Model
from A2net import A2net
import numpy as np
import scipy.signal as signal
from result import subjective_evaluation, objective_evaluation
from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM
from tqdm import tqdm
import argparse
import pickle
from evaluation import wer, snr, lsd
from pesq import pesq_batch
from torch.cuda.amp import GradScaler, autocast

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
Loss = nn.L1Loss()

seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32
rate_mic = 16000
rate_imu = 1600
length = 5
stride = 4

freq_bin_high = 8 * (int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1)
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(length * rate_mic/(seg_len_mic-overlap_mic)) + 1
freq_bin_limit = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1



def sample_evaluation(x, noise, y, audio_only=False):
    x = x.to(device=device, dtype=torch.float)
    magnitude = torch.abs(noise).to(device=device, dtype=torch.float)
    phase = torch.angle(noise).to(device=device, dtype=torch.float)
    noise_real = noise.real.to(device=device, dtype=torch.float)
    noise_imag = noise.imag.to(device=device, dtype=torch.float)
    y = y.to(device=device).squeeze(1)
    if audio_only:
        predict1 = model(magnitude)
    else:
        predict1, predict2 = model(x, magnitude)

    # either predict the spectrogram, or predict the CIRM
    #predict1 = torch.exp(1j * phase[:, :freq_bin_high, :]) * predict1

    cRM = decompress_cIRM(predict1.permute(0, 2, 3, 1))

    enhanced_real = cRM[..., 0] * noise_real.squeeze(1) - cRM[..., 1] * noise_imag.squeeze(1)
    enhanced_imag = cRM[..., 1] * noise_real.squeeze(1) + cRM[..., 0] * noise_imag.squeeze(1)
    predict1 = torch.complex(enhanced_real, enhanced_imag)

    predict = predict1.cpu().numpy()
    predict = np.pad(predict, ((0, 0), (0, int(seg_len_mic / 2) + 1 - freq_bin_high), (0, 0)))
    _, predict = signal.istft(predict, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

    y = y.cpu().numpy()
    y = np.pad(y, ((0, 0), (0, int(seg_len_mic / 2) + 1 - freq_bin_high), (0, 0)))
    _, y = signal.istft(y, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

    return np.stack([np.array(pesq_batch(16000, y, predict, 'wb', n_processor=0, on_error=1)), snr(y, predict), lsd(y, predict)], axis=1)
def sample(x, noise, y, audio_only=False):
    cIRM = build_complex_ideal_ratio_mask(noise.real, noise.imag, y.real, y.imag)  # [B, 2, F, T]
    # cIRM = cIRM.to(device=device, dtype=torch.float)
    # x = x.to(device=device, dtype=torch.float)
    # noise = torch.abs(noise).to(device=device, dtype=torch.float)
    # y = y.abs().to(device=device, dtype=torch.float)

    noise = noise.abs()
    y = y.abs()

    if audio_only:
        with autocast():
            predict1 = model(noise)
        loss = Loss(predict1, cIRM)
    else:
        predict1, predict2 = model(x, noise)
        loss1 = Loss(predict1, y)
        loss2 = Loss(predict2, y[:, :, :33, :])
        loss = loss1 + 0.05 * loss2
    return loss

def train(dataset, EPOCH, lr, BATCH_SIZE, model, save_all=False):
    length = len(dataset)
    test_size = min(int(0.1 * length), 2000)
    train_size = length - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = Data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scaler = GradScaler()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    loss_best = 1
    loss_curve = []
    ckpt_best = model.state_dict()
    for e in range(EPOCH):
        Loss_list = []
        for i, (x, noise, y) in enumerate(tqdm(train_loader)):
            loss = sample(x, noise, y, audio_only=True)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            Loss_list.append(loss.item())
            if i % 300 == 0:
                print("epoch: ", e, "iteration: ", i, "training loss: ", loss.item())
        mean_lost = np.mean(Loss_list)
        loss_curve.append(mean_lost)
        Metric = []
        with torch.no_grad():
            for x, noise, y in test_loader:
                metric = sample_evaluation(x, noise, y, audio_only=True)
                Metric.append(metric)
        avg_metric = np.mean(np.concatenate(Metric, axis=0), axis=0)
        print(avg_metric)
        scheduler.step()


        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            if save_all:
                torch.save(ckpt_best, 'pretrain/' + str(loss_curve[-1]) + '.pth')
    return ckpt_best, loss_curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-pre train, 1-main benchmark, 2-mirco benchmark')
    args = parser.parse_args()

    if args.mode == 0:
        BATCH_SIZE = 32
        lr = 0.001
        EPOCH = 40
        dataset = NoisyCleanSet(['json/train.json', 'json/all_noise.json'], simulation=True, ratio=1)

        #model = nn.DataParallel(A2net(), device_ids=[0]).to(device)
        model = Model(num_freqs=264).cuda()
        #model = nn.DataParallel(Model(num_freqs=264), device_ids=[0, 1]).to(device)
        ckpt_best, loss_curve = train(dataset, EPOCH, lr, BATCH_SIZE, model, save_all=True)

        plt.plot(loss_curve)
        plt.savefig('loss.png')

    elif args.mode == 1:
        # synthetic dataset
        file = open(pkl_folder + "clean_train_paras.pkl", "rb")
        norm_clean = pickle.load(file)
        source = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        datasets = []
        for target in source:
            datasets.append(
                IMUSPEECHSet('json/clean_train_imuexp7.json', 'json/clean_train_wavexp7.json', 'json/speech.json',
                             person=[target], minmax=norm_clean[target]))
        train_dataset = Data.ConcatDataset(datasets)

        model = nn.DataParallel(A2net()).to(device)
        ckpt = torch.load('pretrain/L1/0.0013439175563689787.pth')
        model.load_state_dict(ckpt)
        ckpt, _ = train(train_dataset, 5, 0.001, 32, Loss, device, model)
        model.load_state_dict(ckpt)
        # change noise type
        for target in source:
            datasets.append(
                IMUSPEECHSet('json/clean_train_imuexp7.json', 'json/clean_train_wavexp7.json', 'json/background.json',
                             person=[target], minmax=norm_clean[target]))
        train_dataset = Data.ConcatDataset(datasets)

        length = len(train_dataset)
        test_size = min(int(0.2 * length), 2000)
        train_size = length - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size],
                                                                    torch.Generator().manual_seed(0))

        # test on synthetic noise
        PESQ, SNR, LSD = objective_evaluation(model, test_dataset)
        mean_PESQ = np.mean(PESQ, axis=0)[0]
        np.savez(str(mean_PESQ) + '.npz', PESQ=PESQ, SNR=SNR, LSD=LSD)

    elif args.mode == 2:
        # train one by one
        file = open(pkl_folder + "clean_train_paras.pkl", "rb")
        norm_clean = pickle.load(file)
        file = open(pkl_folder + "noise_train_paras.pkl", "rb")
        norm_noise = pickle.load(file)

        source = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        #source = ["hou", "he"]
        candidate = ["he", "shi", "hou", "1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai"]
        #candidate = ['hou']
        for target in candidate:
            source = [x for x in source if x != target]
            datasets = []
            for c in source:
                datasets.append(IMUSPEECHSet('json/noise_train_imu.json', 'json/noise_train_gt.json', 'json/noise_train_wav.json', simulate=False, person=[c], norm=norm_noise[c]))
            train_dataset = Data.ConcatDataset(datasets)
            user_dataset = IMUSPEECHSet('json/train_imu.json', 'json/train_wav.json', 'json/train_wav.json', ratio=1, person=[target], norm=norm_clean[target])
            model = nn.DataParallel(A2net()).to(device)
            #ckpt = torch.load('pretrain/L1/0.0013439175563689787.pth')
            #ckpt = torch.load('pretrain/mel/0.0034707123340922408.pth')
            ckpt = torch.load('pretrain/0.06400973408017308.pth')
            # for f in os.listdir('checkpoint/5min'):
            #     if f[-3:] == 'pth' and f[:len(target)] == target:
            #         pth_file = f
            # ckpt = torch.load('checkpoint/5min/' + pth_file)
            # ckpt = {'module.' + k: v for k, v in ckpt.items()}
            # print(pth_file)
            model.load_state_dict(ckpt)

            # ckpt, _ = train(train_dataset, 5, 0.001, 32, Loss, device, model)
            # model.load_state_dict(ckpt)
            # ckpt, _ = train(user_dataset, 2, 0.0001, 4, Loss, device, model)
            # model.load_state_dict(ckpt)

            file = open(pkl_folder + "noise_paras.pkl", "rb")
            paras = pickle.load(file)
            dataset = IMUSPEECHSet('json/noise_imu.json', 'json/noise_gt.json', 'json/noise_wav.json', person=[target], simulate=False, phase=True, norm=paras[target])
            WER = subjective_evaluation(model, dataset)
            mean_WER = np.mean(WER)
            np.savez(target + '_' + str(mean_WER) + '.npz', PESQ=PESQ, SNR=SNR, WER=WER)
            print(target, mean_WER)



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






