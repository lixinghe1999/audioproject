import torch
import torch.utils.data as Data
from data import NoisyCleanSet, IMUSPEECHSet, transfer_function_generator, read_transfer_function, weighted_loss
from unet import UNet
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
if __name__ == "__main__":

    EPOCH = 10
    BATCH_SIZE = 1
    model = UNet(1, 1)
    # because the training is done on multiple cards
    # model = torch.nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load("checkpoints/train_0_0.02572406893459094.pth"))
    #model.load_state_dict(torch.load("checkpoint_5_0.005457089898101558.pth"))

    transfer_function, variance, noise = read_transfer_function('transfer_function')
    transfer_function = transfer_function_generator(transfer_function)
    variance = transfer_function_generator(variance)

    train_dataset1 = NoisyCleanSet(transfer_function, variance, noise,'speech100.json', alpha=(6, 0.012, 0.0583))
    test_dataset = NoisyCleanSet(transfer_function, variance, noise,'devclean.json', alpha=(6, 0.012, 0.0583))
    # train_dataset2 = NoisyCleanSet(transfer_function, hist, bins, noise_hist, noise_bin, 'speech360.json')
    # train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
    #IMU_dataset = IMUSPEECHSet('imuexp4.json', 'wavexp4.json', minmax=(0.012, 0.01))
    IMU_dataset = IMUSPEECHSet('imuexp6.json', 'wavexp6.json', minmax=(0.012, 0.002))

    # train_loader = Data.DataLoader(dataset=train_dataset1, num_workers=4, batch_size=BATCH_SIZE, shuffle=True,
    #                                pin_memory=True)
    #test_dataset = NoisyCleanSet(transfer_function, variance, hist, bins, noise_hist, noise_bins, noise, 'devclean.json')
    test_loader = Data.DataLoader(dataset=IMU_dataset, num_workers=4, batch_size=1, shuffle=True)
    Loss = nn.SmoothL1Loss(beta=0.1)
    #Loss = nn.SmoothL1Loss(beta=0.0001)
    #Loss = nn.HuberLoss(delta=50)
    #ssim_loss = MS_SSIM(data_range=1, win_size=5, channel=1, size_average=True)
    with torch.no_grad():
        i = 0
        for x, y in test_loader:
            x, y = x.to(dtype=torch.float), y.to(dtype=torch.float)
            predict = model(x)
            #print(Loss(x, y), Loss(predict, y))
            print(weighted_loss(x, y), weighted_loss(predict, y))
            fig, axs = plt.subplots(3)
            axs[0].imshow(x[0, 0, :, :], aspect='auto')
            axs[1].imshow(y[0, 0, :, :], aspect='auto')
            axs[2].imshow(predict[0, 0, :, :], aspect='auto')
            plt.show()