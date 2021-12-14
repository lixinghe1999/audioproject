import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from data import NoisyCleanSet, read_transfer_function, transfer_function_generator, IMUSPEECHSet, weighted_loss
from unet import UNet
import numpy as np
from tqdm import tqdm

# IMU data 0.012 or 0.011
# SPEECH 0.01 -- ratio 6
# LibriSPEECH 0.0583 divided by 6 and add noise

# max for IMU exp4 = 0.01200766
# max for SPEECH exp4 = 0.0097796 ratio = 5.93
# max for IMU exp6 = 0.01093818
# max for SPEECH exp6 = 0.001848658 ratio = 31.53
# max for Librispeech 0.058281441500849615
if __name__ == "__main__":

    EPOCH = 50
    BATCH_SIZE = 32
    lr = 0.03
    pad = True
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model = UNet(1, 1).to(device)
    #model.load_state_dict(torch.load("checkpoint_41_0.002247016663086074.pth"))

    # # only select weight we want
    # pretrained_dict = torch.load("pretrained.pth")
    # model_dict = model.state_dict()
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)

    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    # model = model.cuda(device=device_ids[0])

    transfer_function, variance, noise = read_transfer_function('transfer_function')
    transfer_function = transfer_function_generator(transfer_function)
    variance = transfer_function_generator(variance)
    train_dataset1 = NoisyCleanSet(transfer_function, variance, noise, 'speech100.json', alpha=(6, 0.012, 0.0583))
    #train_dataset2 = NoisyCleanSet(transfer_function, variance, hist, bins, noise_hist, noise_bins, noise, 'speech360.json')
    #train_dataset = IMUSPEECHSet('imuexp4.json', 'wavexp4.json', minmax=(0.01200766, 0.0097796))
    #train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])

    test_dataset = NoisyCleanSet(transfer_function, variance, noise, 'devclean.json', alpha=(6, 0.012, 0.0583))

    train_loader = Data.DataLoader(dataset=train_dataset1, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.00001)
    #Loss = nn.MSELoss()
    #Loss = nn.SmoothL1Loss(beta=0.1)
    #Loss = nn.L1Loss()
    #ssim_loss = MS_SSIM(data_range=1, win_size=5, channel=1, size_average=True)
    for epoch in range(EPOCH):
        for x, y in tqdm(train_loader):
            x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
            predict = model(x)
            loss = weighted_loss(predict, y, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            Loss_all = []
            for x, y in test_loader:
                x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
                predict = model(x)
                loss = weighted_loss(predict, y, device)
                Loss_all.append(loss.item())
            val_loss = np.mean(Loss_all)
        scheduler.step()
        torch.save(model.state_dict(), 'checkpoint' + '_' + str(epoch) + '_' + str(val_loss.item()) + '.pth')
    print('________________________________________')
    print('finish training')