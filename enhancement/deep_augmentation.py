import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from dataset import NoisyCleanSet
from SEANet import SEANet_mapping
from evaluation import SI_SDR
import os
'''
this script describe the deep learning mapping proposed by SEANet(google)
'''
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
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
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
            loss = nn.functional.l1_loss(predict, acc).item()
            si_sdr = SI_SDR(acc.cpu().numpy(), predict.cpu().numpy())
            print(si_sdr)
            print(loss, torch.max(predict), torch.max(acc))
            sdr_list.append(si_sdr)

if __name__ == '__main__':
    model = SEANet_mapping()
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # This script is for model pre-training on LibriSpeech
    BATCH_SIZE = 64
    lr = 0.001
    EPOCH = 10
    people = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
    dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'], time_domain=True,
                            person=people, simulation=True)
    model = model.to(device)

    ckpt_best, loss_curve = train(dataset, EPOCH, lr, BATCH_SIZE, model, save_all=True)

    # For the testing: Si-SDR
    # ckpt_dir = 'pretrain/deep_augmentation'
    # ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[0]
    # print("load checkpoint: {}".format(ckpt_name))
    # ckpt = torch.load(ckpt_name)
    # model.load_state_dict(ckpt)
    # test(dataset, BATCH_SIZE, model)