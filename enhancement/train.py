import os
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)

from dataset import NoisyCleanSet, EMSBDataset
import json

from fullsubnet import fullsubnet
from new_vibvoice import vibvoice
from conformer import TSCNet
from SEANet import SEANet
from voicefilter import voicefilter
from sudormrf import sudormrf

import numpy as np
from tqdm import tqdm
import argparse
from discriminator import Discriminator_time, Discriminator_spectrogram, MultiScaleDiscriminator, SingleScaleDiscriminator
import model_zoo
def parse_sample(sample, text=False):
    if text:
        data, text = sample[:-1], sample[-1]
    else:
        data = sample
        text = None
    if len(data) == 3:
        clean, noise, acc = data
    else:
        clean, noise = data
        acc = None
    return text, clean, noise, acc

def inference(dataset, BATCH_SIZE, model, text=False):
    text_inference = text
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    Metric = []
    with torch.no_grad():
        for sample in test_loader:
            text, clean, noise, acc = parse_sample(sample, text=text_inference)
            metric = getattr(model_zoo, 'test_' + model_name)(model, acc, noise, clean, device, text)
            Metric.append(metric)
    avg_metric = np.mean(np.concatenate(Metric, axis=0), axis=0)
    return avg_metric

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

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    if discriminator is not None:
        optimizer_disc = torch.optim.AdamW(params=discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
        #scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer_disc, step_size=5, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    loss_best = 100
    loss_curve = []
    ckpt_best = model.state_dict()
    for e in range(EPOCH):
        Loss_list = []
        for i, sample in enumerate(tqdm(train_loader)):
            text, clean, noise, acc = parse_sample(sample)
            loss = getattr(model_zoo, 'train_' + model_name)(model, acc, noise, clean, optimizer, device)
            if i % 500 == 0:
                print(loss)
            Loss_list.append(loss)
        mean_lost = np.mean(Loss_list)
        loss_curve.append(mean_lost)
        scheduler.step()
        avg_metric = inference(test_dataset, 8, model)
        print(avg_metric)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            metric_best = avg_metric
        if save_all:
            torch.save(ckpt_best, 'pretrain/' + str(mean_lost) + '.pth')
    torch.save(ckpt_best, 'pretrain/' + str(metric_best) + '.pth')
    return ckpt_best, loss_curve, metric_best



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of the script')
    parser.add_argument('--model', action="store", type=str, default='vibvoice', required=False,
                        help='choose the model')
    args = parser.parse_args()
    torch.cuda.set_device(1)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # select available model from vibvoice, fullsubnet, conformer,
    model_name = args.model
    model = globals()[model_name]().to(device)
    people = ["hou", "1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he"]

    # discriminator = MultiScaleDiscriminator().to(device)
    # model = torch.nn.DataParallel(model, device_ids=[1])

    if args.mode == 0:
        # This script is for model pre-training on LibriSpeech
        BATCH_SIZE = 8
        lr = 0.001
        EPOCH = 20
        # ckpt = torch.load('voiceSplit-trained-with-MSE-GE2E-Seungwonpark-best_checkpoint.pt')
        # model.load_state_dict(ckpt['model'])

        dataset = NoisyCleanSet(['json/librispeech-100.json', 'json/tr.json'], simulation=True,
                                ratio=1, rir='json/rir.json', dvector=None)
        test_dataset = NoisyCleanSet(['json/librispeech-dev.json', 'json/cv.json'], simulation=True,
                                ratio=1, rir='json/rir.json', dvector=None)

        ckpt_best, loss_curve, metric_best = train([dataset, test_dataset], EPOCH, lr, BATCH_SIZE, model, discriminator=None, save_all=True)
        plt.plot(loss_curve)
        plt.savefig('loss.png')
    elif args.mode == 1:
        # This script is for model fine-tune on self-collected dataset, by default-with all noises
        people = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        BATCH_SIZE = 16
        lr = 0.0001
        EPOCH = 10
        dvector = 'spk_embedding/our'
        ckpt_dir = 'pretrain/voicefilter'
        ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[-1]
        ckpt_name = 'pretrain/fullsubnet_our.pth'
        print("load checkpoint: {}".format(ckpt_name))
        ckpt_start = torch.load(ckpt_name)
        model.load_state_dict(ckpt_start)

        # checkpoint = torch.load("fullsubnet_best_model_58epochs.tar")
        # print('loading pre-trained FullSubNet (SOTA)', checkpoint['best_score'])
        # model.load_state_dict(checkpoint['model'])

        # checkpoint = torch.load("fullsubnet_best_model_58epochs.tar")
        # model.load_state_dict(checkpoint['model'])

        train_dataset = NoisyCleanSet(['json/train_gt.json', 'json/cv.json', 'json/train_imu.json'],
                                        simulation=True, person=people, ratio=0.8, dvector=dvector)
        test_dataset = NoisyCleanSet(['json/train_gt.json', 'json/cv.json', 'json/train_imu.json'],
                                          simulation=True, person=people, ratio=-0.2, dvector=dvector)

        # positions = ['glasses', 'vr-up', 'vr-down', 'headphone-inside', 'headphone-outside', 'cheek', 'temple', 'back', 'nose']
        # train_dataset2 = NoisyCleanSet(['json/position_gt.json', 'json/cv.json', 'json/position_imu.json'],
        #                                simulation=True, person=positions, ratio=r,)
        # test_dataset2 = NoisyCleanSet(['json/position_gt.json', 'json/cv.json', 'json/position_imu.json'],
        #                               simulation=True, person=positions, ratio=-0.2,)
        #
        # train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset2])
        # test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset2])


        ckpt, loss_curve, metric_best = train([train_dataset, test_dataset], EPOCH, lr, BATCH_SIZE, model, discriminator=None)
        model.load_state_dict(ckpt)

        for p in ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/cv.json', 'json/train_imu.json'],
                                    person=[p], simulation=True, ratio=-0.2, dvector=dvector)
            avg_metric = inference(dataset, 4, model)
            print(p, avg_metric)

        for noise in ['background.json', 'librispeech-dev.json', 'music.json']:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/' + noise,  'json/train_imu.json'],
                                    person=people, simulation=True, ratio=-0.2, dvector=dvector)
            avg_metric = inference(dataset, 4, model)
            print(noise, avg_metric)

        for level in [10, 5, 1]:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/cv.json',  'json/train_imu.json'], person=people,
                                     simulation=True, snr=[level - 1, level + 1], ratio=-0.2, dvector=dvector)
            avg_metric = inference(dataset, 4, model)
            print(level, avg_metric)

        positions = ['glasses', 'vr-up', 'vr-down', 'headphone-inside', 'headphone-outside', 'cheek', 'temple', 'back', 'nose']
        for p in positions:
            dataset = NoisyCleanSet(['json/position_gt.json', 'json/cv.json', 'json/position_imu.json'],
                                    person=[p], simulation=True, ratio=-0.2, dvector=dvector)
            avg_metric = inference(dataset, 4, model)
            print(p, avg_metric)
    elif args.mode == 2:
        # evaluation for WER
        checkpoint = torch.load("fullsubnet_best_model_58epochs.tar")
        print('loading pre-trained FullSubNet (SOTA)', checkpoint['best_score'])
        model.load_state_dict(checkpoint['model'])

        # ckpt_dir = 'pretrain/vibvoice'
        # ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[-1]
        # print('loaded checkpoint:', ckpt_name)
        # ckpt_start = torch.load(ckpt_name)

        # ckpts = []
        # for p in ['he']:
        #     model.load_state_dict(ckpt_start)
        #     p_except = [i for i in people if i != p]
        #     train_dataset = NoisyCleanSet(['json/noise_train_gt.json', 'json/noise_train_wav.json', 'json/noise_train_imu.json'],
        #                                  person=p_except, simulation=False, text=False)
        #     ckpt, _, _ = train(train_dataset, 1, 0.0001, 16, model)
        #
        #     train_dataset = NoisyCleanSet(['json/train_gt.json', 'json/dev.json', 'json/train_imu.json'],
        #                                   person=[p], time_domain=time_domain, simulation=True,
        #                                   rir=None, text=False, snr=(0, 20))
        #     ckpt, _, _ = train(train_dataset, 2, 0.0001, 16, model)
        #    ckpts.append(ckpt)
        for p in people:
            #model.load_state_dict(ckpt)
            test_dataset = NoisyCleanSet(['json/noise_gt.json', 'json/noise_wav.json', 'json/noise_imu.json'],
                                         person=[p], simulation=False, text=True)
            avg_metric = inference(test_dataset, 4, model, text=True)
            print(p, avg_metric)
        for env in ['airpod', 'freebud', 'galaxy', 'office', 'corridor', 'stair', 'human-corridor', 'human-hall', 'human-outdoor']:
            test_dataset = NoisyCleanSet(['json/noise_gt.json', 'json/noise_wav.json', 'json/noise_imu.json'],
                                         person=[env], simulation=False, text=True)
            avg_metric = inference(test_dataset, 4, model, text=True)
            print(env, avg_metric)
        #
        test_dataset = NoisyCleanSet(['json/mobile_gt.json', 'json/mobile_wav.json', 'json/mobile_imu.json'],
                                     person=['he'], simulation=False, text=True)
        avg_metric = inference(test_dataset, 4, model, text=True)
        print('mobile result', avg_metric)
    else:
        # investigate the performance of FullSubnet
        checkpoint = torch.load("fullsubnet_best_model_58epochs.tar")
        print('loading pre-trained FullSubNet (SOTA)', checkpoint['best_score'])
        model.load_state_dict(checkpoint['model'])

        dataset = NoisyCleanSet(['json/DNSclean.json', 'json/DNSnoisy.json'],
                                simulation=False, ratio=1)
        avg_metric = inference(dataset, 4, model)
        print(avg_metric)

        # dataset = NoisyCleanSet(['json/dev.json', 'json/cv.json'],
        #                            simulation=True, ratio=0.1, rir='json/rir.json')
        # avg_metric = inference(dataset, 4, model)
        # print(avg_metric)
