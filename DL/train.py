import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from data import NoisyCleanSet, read_transfer_function, transfer_function_generator, IMUSPEECHSet, weighted_loss
from unet import UNet, TinyUNet
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

# for mel spectrogram
# max for IMU 0.28
# max for SPEECH 0.73

# for spectrogram
# max for IMU 0.0132
# max for SPEECH 0.0037

# for Librispeech
# 0.0637 ratio = 17.2
if __name__ == "__main__":

    EPOCH = 25
    BATCH_SIZE = 16
    lr = 0.0025
    pad = True
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load("checkpoints/pretrain_0.011717716008242146.pth"))
    #model.load_state_dict(torch.load("checkpoint_1_0.012661107610363294.pth"))
    #model = TinyUNet(1, 1).to(device)


    # # only select weight we want
    # pretrained_dict = torch.load("pretrained.pth")
    # model_dict = model.state_dict()
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)

    # transfer_function, variance = read_transfer_function('../iterated_function')
    # train_dataset1 = NoisyCleanSet(transfer_function, variance, 'speech100.json', alpha=(17.2, 0.07, 0.063))
    # test_dataset1 = NoisyCleanSet(transfer_function, variance, 'devclean.json', alpha=(17.2, 0.07, 0.063))

    train_dataset2 = IMUSPEECHSet('clean_imuexp6.json', 'clean_wavexp6.json', minmax=(0.0132, 0.0037))
    length = len(train_dataset2)
    train_size, validate_size = int(0.8 * length), int(0.2 * length)
    train_dataset2, test_dataset2 = torch.utils.data.random_split(train_dataset2, [train_size, validate_size], torch.Generator().manual_seed(0))

    train_loader = Data.DataLoader(dataset=train_dataset2, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_dataset2, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    down1_params = list(map(id, model.down1.parameters()))
    down2_params = list(map(id, model.down2.parameters()))
    down3_params = list(map(id, model.down3.parameters()))
    down4_params = list(map(id, model.down4.parameters()))
    base_params = filter(lambda p: id(p) not in down4_params + down3_params + down2_params + down1_params,
                         model.parameters())
    optimizer = torch.optim.AdamW([{'params': base_params}, {'params': model.down1.parameters(), 'lr': lr*0.1}, {'params': model.down2.parameters(), 'lr': lr*0.2},
                                   {'params': model.down3.parameters(), 'lr': lr*0.3},{'params': model.down4.parameters(), 'lr': lr*0.4}], lr=lr, weight_decay=0.05)

    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.00001)
    Loss = nn.L1Loss()
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
                loss = Loss(predict, y)
                Loss_all.append(loss.item())
            val_loss = np.mean(Loss_all)
        scheduler.step()
        torch.save(model.state_dict(), 'checkpoint' + '_' + str(epoch) + '_' + str(val_loss.item()) + '.pth')
    print('________________________________________')
    print('finish training')