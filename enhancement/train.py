import os
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)

from dataset import NoisyCleanSet, EMSBDataset
import json

from fullsubnet import FullSubNet
from new_vibvoice import A2net
from conformer import TSCNet
from SEANet import SEANet

import numpy as np
from tqdm import tqdm
import argparse
from discriminator import Discriminator_time, Discriminator_spectrogram, MultiScaleDiscriminator, SingleScaleDiscriminator
from model_zoo import train_SEANet, test_SEANet, train_vibvoice, test_vibvoice, train_fullsubnet, test_fullsubnet, \
    train_conformer, test_conformer


def train(dataset, EPOCH, lr, BATCH_SIZE, model, discriminator=None, save_all=False):
    if isinstance(dataset, list):
        # with pre-defined train/ test
        train_dataset, test_dataset = dataset
    else:
        # without pre-defined train/ test
        length = len(dataset)
        test_size = min(int(0.1 * length), 1000)
        train_size = length - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                                   pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    if discriminator is not None:
        optimizer_disc = torch.optim.AdamW(params=discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
        #scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer_disc, step_size=5, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    loss_best = 100
    loss_curve = []
    ckpt_best = model.state_dict()
    for e in range(EPOCH):
        Loss_list = []
        for i, (acc, noise, clean) in enumerate(tqdm(train_loader)):
            #loss, discrim_loss = train_SEANet(model, acc, noise, clean, optimizer, optimizer_disc, discriminator, device)
            loss = train_vibvoice(model, acc, noise, clean, optimizer, device)
            Loss_list.append(loss)
        mean_lost = np.mean(Loss_list)
        loss_curve.append(mean_lost)
        scheduler.step()
        Metric = []
        with torch.no_grad():
            for acc, noise, clean in tqdm(test_loader):
                metric = test_vibvoice(model, acc, noise, clean, device)
                Metric.append(metric)
        avg_metric = np.mean(np.concatenate(Metric, axis=0), axis=0)
        print(avg_metric, mean_lost)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            metric_best = avg_metric
        if save_all:
            torch.save(ckpt_best, 'pretrain/' + str(mean_lost) + '.pth')
    torch.save(ckpt_best, 'pretrain/' + str(metric_best) + '.pth')
    return ckpt_best, loss_curve, metric_best

def inference(dataset, BATCH_SIZE, model):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    Metric = []
    with torch.no_grad():
        for data in test_loader:
            if len(data) == 3:
                acc, noise, clean = data
                metric = test_fullsubnet(model, acc, noise, clean, device)
            else:
                text, acc, noise, clean = data
                metric = test_fullsubnet(model, acc, noise, clean, device, text)
            Metric.append(metric)
    Metric = np.concatenate(Metric, axis=0)
    return Metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-pre train, 1-main benchmark, 2-mirco benchmark')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # model = A2net().to(device)
    model = FullSubNet(num_freqs=257, num_groups_in_drop_band=1).to(device)
    # model = SEANet().to(device)

    #discriminator = MultiScaleDiscriminator().to(device)
    time_domain = False

    # model = torch.nn.DataParallel(model, device_ids=[1])
    # discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1])
    # discriminator = Discriminator_spectrogram().to(device)

    if args.mode == 0:
        # This script is for model pre-training on LibriSpeech
        BATCH_SIZE = 128
        lr = 0.0001
        EPOCH = 30
        ckpt_start = torch.load('pretrain/0.46906685339063947.pth')
        model.load_state_dict(ckpt_start)

        dataset = NoisyCleanSet(['json/train.json', 'json/tr.json'], time_domain=time_domain, simulation=True,
                                ratio=1, rir='json/rir_noise.json')
        # with open('json/EMSB.json', 'r') as f:
        #     data = json.load(f)
        #     person = data.keys()
        # dataset = EMSBDataset(['json/EMSB.json', 'json/all_noise.json'], time_domain=time_domain, simulation=True,
        #                         ratio=0.6, person=person)

        ckpt_best, loss_curve, metric_best = train(dataset, EPOCH, lr, BATCH_SIZE, model, discriminator=None, save_all=True)
        plt.plot(loss_curve)
        plt.savefig('loss.png')
    elif args.mode == 1:
        # This script is for model fine-tune on self-collected dataset, by default-with all noises
        people = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        BATCH_SIZE = 16
        lr = 0.0001
        EPOCH = 10
        r = 0.8

        # ckpt_dir = 'pretrain/vibvoice_rir'
        # ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[-1]
        # ckpt_name = 'pretrain/0.4294044267669987.pth'
        # print("load checkpoint: {}".format(ckpt_name))
        # ckpt_start = torch.load(ckpt_name)
        # model.load_state_dict(ckpt_start)

        # Load pre-trained FullSubnet
        checkpoint = torch.load("fullsubnet_best_model_58epochs.tar")
        print(checkpoint['best_score'])
        model.load_state_dict(checkpoint['model'])

        #
        # train_dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'],
        #                                 time_domain=time_domain, simulation=True, person=people, ratio=r,)
        # test_dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'],
        #                                   time_domain=time_domain, simulation=True, person=people, ratio=-0.2,)

        # extra dataset for other positions
        # positions = ['glasses', 'vr-up', 'vr-down', 'headphone-inside', 'headphone-outside', 'cheek', 'temple', 'back', 'nose']
        # train_dataset2 = NoisyCleanSet(['json/position_gt.json', 'json/all_noise.json', 'json/position_imu.json'],
        #                                time_domain=time_domain, simulation=True, person=positions, ratio=r,)
        # test_dataset2 = NoisyCleanSet(['json/position_gt.json', 'json/all_noise.json', 'json/position_imu.json'],
        #                               time_domain=time_domain, simulation=True, person=positions, ratio=-0.2,)

        # train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset2])
        # test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset2])


        # ckpt, loss_curve, metric_best = train([train_dataset, test_dataset], EPOCH, lr, BATCH_SIZE, model, discriminator=None)
        # model.load_state_dict(ckpt)
        # Optional Micro-benchmark

        rirs = ['json/smallroom.json', 'json/mediumroom.json', 'json/largeroom.json']
        for rir in rirs:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'],
                                    person=people, time_domain=time_domain, simulation=True, ratio=-0.2,)
            Metric = inference(dataset, BATCH_SIZE, model)
            avg_metric = np.mean(Metric, axis=0)
            print(rir, avg_metric)

        for p in ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'],
                                    person=people, time_domain=time_domain, simulation=True, ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model)
            avg_metric = np.mean(Metric, axis=0)
            print(p, avg_metric)

        for noise in ['background.json', 'dev.json', 'music.json']:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/' + noise,  'json/train_imu.json'],
                                    person=people, time_domain=time_domain, simulation=True, ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model)
            avg_metric = np.mean(Metric, axis=0)
            print(noise, avg_metric)

        for level in [5, 1, -3]:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json',  'json/train_imu.json'], person=people,
                                    time_domain=time_domain, simulation=True, snr=[level - 1, level + 1], ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model)
            avg_metric = np.mean(Metric, axis=0)
            print(level, avg_metric)

        positions = ['glasses', 'vr-up', 'vr-down', 'headphone-inside', 'headphone-outside', 'cheek', 'temple', 'back', 'nose']
        for p in positions:
            dataset = NoisyCleanSet(['json/position_gt.json', 'json/all_noise.json', 'json/position_imu.json'],
                                    person=[p], time_domain=time_domain, simulation=True, ratio=-0.2)
            Metric = inference(dataset, BATCH_SIZE, model)
            avg_metric = np.mean(Metric, axis=0)
            print(p, avg_metric)
    elif args.mode == 2:
        # evaluation for WER
        ckpt_dir = 'pretrain/vibvoice'
        ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[-1]
        ckpt_name = 'pretrain/vibvoice_rir/[ 2.52787821 14.42417432  3.50193285].pth'
        print('loaded checkpoint:', ckpt_name)
        ckpt_start = torch.load(ckpt_name)
        people = ["hou", "1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he"]
        ckpts = []
        for p in ['he']:
            model.load_state_dict(ckpt_start)
            p_except = [i for i in people if i != p]
            train_dataset = NoisyCleanSet(['json/noise_train_gt.json', 'json/noise_train_wav.json', 'json/noise_train_imu.json'],
                                         person=p_except, time_domain=time_domain, simulation=False, text=False)
            ckpt, _, _ = train(train_dataset, 1, 0.0001, 16, model)
        #
        #     train_dataset = NoisyCleanSet(['json/train_gt.json', 'json/dev.json', 'json/train_imu.json'],
        #                                   person=[p], time_domain=time_domain, simulation=True,
        #                                   rir=None, text=False, snr=(0, 20))
        #     ckpt, _, _ = train(train_dataset, 2, 0.0001, 16, model)
            ckpts.append(ckpt)
        for ckpt, p in zip(ckpts,  ['he']):
            model.load_state_dict(ckpt)
            test_dataset = NoisyCleanSet(['json/noise_gt.json', 'json/noise_wav.json', 'json/noise_imu.json'],
                                         person=[p], time_domain=time_domain, simulation=False, text=True)
            metric = inference(test_dataset, 4, model)
            avg_metric = np.mean(metric, axis=0)
            print(p, avg_metric)
        for env in ['airpod', 'freebud', 'galaxy', 'office', 'corridor', 'stair', 'human-corridor', 'human-hall', 'human-outdoor']:
            test_dataset = NoisyCleanSet(['json/noise_gt.json', 'json/noise_wav.json', 'json/noise_imu.json'],
                                         person=[env], time_domain=time_domain, simulation=False, text=True)
            metric = inference(test_dataset, 4, model)
            avg_metric = np.mean(metric, axis=0)
            print(env, avg_metric)
        #
        test_dataset = NoisyCleanSet(['json/mobile_gt.json', 'json/mobile_wav.json', 'json/mobile_imu.json'],
                                     person=['he'], time_domain=time_domain, simulation=False, text=True)
        metric = inference(test_dataset, 8, model)
        avg_metric = np.mean(metric, axis=0)
        print('mobile result', avg_metric)
