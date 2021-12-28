import torch
import torch.utils.data as Data
from data import NoisyCleanSet, IMUSPEECHSet, transfer_function_generator, read_transfer_function, weighted_loss
from unet import UNet
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
if __name__ == "__main__":
    BATCH_SIZE = 32
    # because the training is done on multiple cards
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load("checkpoints/finetune_0.034651425838166355.pth"))

    transfer_function, variance, noise = read_transfer_function('transfer_function')
    transfer_function = transfer_function_generator(transfer_function)
    variance = transfer_function_generator(variance)

    train_dataset1 = NoisyCleanSet(transfer_function, variance, noise, 'speech100.json', alpha=(6, 0.012, 0.0583))
    test_dataset1 = NoisyCleanSet(transfer_function, variance, noise, 'devclean.json', alpha=(6, 0.012, 0.0583))

    train_dataset2 = IMUSPEECHSet('clean_imuexp6.json', 'clean_wavexp6.json', minmax=(0.012, 0.002))
    test_loader = Data.DataLoader(dataset=train_dataset2, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    Loss = nn.L1Loss()
    with torch.no_grad():
        L = []
        for x, y in test_loader:
            x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
            predict = model(x)
            L.append(Loss(predict, y).item())
        print(np.mean(L))

            #print(Loss(x, y), Loss(predict, y))
            # print(weighted_loss(x, y), weighted_loss(predict, y))
            # fig, axs = plt.subplots(3)
            # axs[0].imshow(x[0, 0, :, :], aspect='auto')
            # axs[1].imshow(y[0, 0, :, :], aspect='auto')
            # axs[2].imshow(predict[0, 0, :, :], aspect='auto')
            # plt.show()