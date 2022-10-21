import os
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
from dataset import NoisyCleanSet
from fullsubnet import FullSubNet
from A2net import A2net
from causal_A2net import Causal_A2net
from conformer import TSCNet
import numpy as np
import scipy.signal as signal
from result import subjective_evaluation, objective_evaluation
from audio_zen.acoustics.feature import drop_band
from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM
from tqdm import tqdm
import argparse
from evaluation import wer, snr, lsd, SI_SDR, batch_pesq
import discriminator


seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32
rate_mic = 16000
rate_imu = 1600


freq_bin_high = 8 * int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1

class STFTLoss(torch.nn.Module):
    """Spectral convergence loss module."""
    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(STFTLoss, self).__init__()
    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        x_mag = torch.clamp(x_mag, min=1e-7)
        y_mag = torch.clamp(y_mag, min=1e-7)
        spectral_convergenge_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        log_stft_magnitude = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        return 0.5 * spectral_convergenge_loss + 0.5 * log_stft_magnitude

def sample_evaluation(model, acc, noise, clean, audio_only=False):
    acc = acc.to(device=device, dtype=torch.float)
    noise_mag = torch.abs(noise).to(device=device, dtype=torch.float)
    noise_pha = torch.angle(noise).to(device=device, dtype=torch.float)
    noise_real = noise.real.to(device=device, dtype=torch.float)
    noise_imag = noise.imag.to(device=device, dtype=torch.float)
    clean = clean.to(device=device).squeeze(1)

    # either predict the spectrogram, or predict the CIRM
    if audio_only:
        predict1 = model(noise_mag)
        cRM = decompress_cIRM(predict1.permute(0, 2, 3, 1))
        enhanced_real = cRM[..., 0] * noise_real.squeeze(1) - cRM[..., 1] * noise_imag.squeeze(1)
        enhanced_imag = cRM[..., 1] * noise_real.squeeze(1) + cRM[..., 0] * noise_imag.squeeze(1)
        predict1 = torch.complex(enhanced_real, enhanced_imag)
    else:
        predict1, _ = model(acc, noise_mag)
        predict1 = torch.exp(1j * noise_pha[:, :, :freq_bin_high, :]) * predict1
        predict1 = predict1.squeeze(1)

    predict = predict1.cpu().numpy()
    predict = np.pad(predict, ((0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
    predict = signal.istft(predict, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)[-1]

    clean = clean.cpu().numpy()
    clean = np.pad(clean, ((0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
    clean = signal.istft(clean, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)[-1]
    metric1 = batch_pesq(clean, predict)
    metric2 = SI_SDR(clean, predict)
    metric3 = lsd(clean, predict)
    return np.stack([metric1, metric2, metric3], axis=1)

def sample(model, acc, noise, clean, optimizer, optimizer_disc=None, discriminator=None, audio_only=False):
    acc = acc.to(device=device, dtype=torch.float)
    noise_mag = noise.abs().to(device=device, dtype=torch.float)
    clean_mag = clean.abs().to(device=device, dtype=torch.float)

    optimizer.zero_grad()
    if audio_only:
        # predict complex Ideal Ratio Mask
        cIRM = build_complex_ideal_ratio_mask(noise.real, noise.imag, clean.real, clean.imag)  # [B, 2, F, T]
        cIRM = cIRM.to(device=device, dtype=torch.float)
        cIRM = drop_band(cIRM, model.module.num_groups_in_drop_band)
        predict1 = model(noise_mag)
        loss = F.l1_loss(predict1, cIRM)

        # predict real and imag
        # noisy_spec = torch.stack([noise.real, noise.imag], 1).to(device=device, dtype=torch.float).permute(0, 1, 3, 2)
        # clean_real, clean_imag = clean.real.to(device=device, dtype=torch.float), clean.imag.to(device=device, dtype=torch.float)
        # est_real, est_imag = model(noisy_spec)
        # est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        # loss = 0.9 * F.mse_loss(est_mag, clean_mag) + 0.1 * F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
    else:
        predict1, _ = model(acc, noise_mag)
        loss = Spectral_Loss(predict1, clean_mag)
        #loss += F.mse_loss(predict2, clean_mag[:, :, :32, :])
    # # adversarial training
    # one_labels = torch.ones(BATCH_SIZE).cuda()
    # predict_fake_metric = discriminator(clean_mag, predict1)
    # gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
    # #print(loss1.item(), loss2.item(), gen_loss_GAN.item())
    # loss += 0.1 * gen_loss_GAN
    loss.backward()
    optimizer.step()
    return loss.item()

    # # discriminator loss
    #
    # predict_audio = predict1.detach().cpu().numpy()
    # predict_audio = np.pad(predict_audio, ((0, 0), (0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
    # predict_audio = signal.istft(predict_audio, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)[-1]
    #
    # clean_audio = np.pad(clean, ((0, 0), (0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
    # clean_audio = signal.istft(clean_audio, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)[-1]
    #
    # pesq_score = discriminator.batch_pesq(clean_audio, predict_audio)
    # # The calculation of PESQ can be None due to silent part
    # if pesq_score is not None:
    #     optimizer_disc.zero_grad()
    #     predict_enhance_metric = discriminator(clean_mag, predict1.detach())
    #     predict_max_metric = discriminator(clean_mag, clean_mag)
    #     discrim_loss = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
    #                           F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
    #     discrim_loss.backward()
    #     optimizer_disc.step()
    # else:
    #     discrim_loss = torch.tensor([0.])
    # return loss.item(), discrim_loss.item()

def train(dataset, EPOCH, lr, BATCH_SIZE, model, discriminator=None, save_all=False, audio_only=False):
    if isinstance(dataset, list):
        # with pre-defined train/ test
        train_dataset, test_dataset = dataset
    else:
        # without pre-defined train/ test
        length = len(dataset)
        test_size = min(int(0.1 * length), 2000)
        train_size = length - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = Data.DataLoader(dataset=train_dataset, num_workers=16, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                                   pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    if discriminator is not None:
        optimizer_disc = torch.optim.AdamW(params=discriminator.parameters(), lr=2 * lr, betas=(0.9, 0.999))
    #optimizer = torch.optim.Adam(params= filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3,)
    loss_best = 1
    loss_curve = []
    ckpt_best = model.state_dict()
    for e in range(EPOCH):
        Loss_list = []
        for i, (acc, noise, clean) in enumerate(tqdm(train_loader)):
            loss = sample(model, acc, noise, clean, optimizer, audio_only=audio_only)
            Loss_list.append(loss)
        mean_lost = np.mean(Loss_list)
        loss_curve.append(mean_lost)
        scheduler.step()
        Metric = []
        with torch.no_grad():
            for acc, noise, clean in tqdm(test_loader):
                metric = sample_evaluation(model, acc, noise, clean, audio_only=audio_only)
                Metric.append(metric)
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

def inference(dataset, BATCH_SIZE, model, audio_only=False):
    test_dataset = dataset
    test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    Metric = []
    with torch.no_grad():
        for x, noise, y in test_loader:
            metric = sample_evaluation(model, x, noise, y, audio_only=audio_only)
            Metric.append(metric)
    Metric = np.concatenate(Metric, axis=0)
    return Metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-pre train, 1-main benchmark, 2-mirco benchmark')
    args = parser.parse_args()
    audio_only = True
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    Spectral_Loss = STFTLoss()
    torch.cuda.set_device(0)
    if args.mode == 0:
        # This script is for model pre-training on LibriSpeech
        BATCH_SIZE = 32
        lr = 0.01
        EPOCH = 30
        dataset = NoisyCleanSet(['json/train.json', 'json/all_noise.json'], simulation=True, ratio=1)

        #model = A2net(inference=False).to(device)
        model = nn.DataParallel(FullSubNet(num_freqs=256, num_groups_in_drop_band=1).to(device), device_ids=[0, 1])
        #model = Causal_A2net(inference=False).to(device)
        #model = TSCNet().to(device)

        # potential ckpt
        # ckpt_dir = 'pretrain/fullsubnet'
        # ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[0]
        # print("load checkpoint: {}".format(ckpt_name))
        # ckpt = torch.load(ckpt_name)
        # model.load_state_dict(ckpt)

        discriminator = discriminator.Discriminator(ndf=16).to(device)
        ckpt_best, loss_curve, metric_best = train(dataset, EPOCH, lr, BATCH_SIZE, model, discriminator,
                                                   save_all=True, audio_only=audio_only)
        plt.plot(loss_curve)
        plt.savefig('loss.png')

    elif args.mode == 1:
        # This script is for model fine-tune on self-collected dataset, by default-with all noises
        BATCH_SIZE = 16
        lr = 0.001
        EPOCH = 10

        ckpt_dir = 'pretrain/causal_vibvoice'
        ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[0]
        print("load checkpoint: {}".format(ckpt_name))
        ckpt = torch.load(ckpt_name)

        #model = A2net(inference=False).to(device)
        #model = FullSubNet(num_freqs=264).to(device)
        model = Causal_A2net(inference=False).to(device)

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
        ckpt, loss_curve, metric_best = train([train_dataset, test_dataset], EPOCH, lr, BATCH_SIZE, model, audio_only=audio_only)

        # Optional Micro-benchmark
        model.load_state_dict(ckpt)
        people = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        for p in people:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'], person=[p], simulation=True, ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model, audio_only=audio_only)
            avg_metric = np.mean(Metric, axis=0)
            print(p, avg_metric)

        for noise in ['background.json', 'dev.json', 'music.json']:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/' + noise,  'json/train_imu.json'], person=people, simulation=True, ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model,  audio_only=audio_only)
            avg_metric = np.mean(Metric, axis=0)
            print(noise, avg_metric)

        for level in [11, 6, 1]:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json',  'json/train_imu.json'], person=people,
                                    simulation=True, snr=[level - 1, level + 1], ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model,  audio_only=audio_only)
            avg_metric = np.mean(Metric, axis=0)
            print(level, avg_metric)

        # Micro-benchmark for different positions
        positions = ['glasses', 'vr-up', 'vr-down', 'headphone-inside', 'headphone-outside', 'cheek', 'temple', 'back', 'nose']
        for p in positions:
            dataset = NoisyCleanSet(['json/position_gt.json', 'json/all_noise.json', 'json/position_imu.json'], person=[p], simulation=True, ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model,  audio_only=audio_only)
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

