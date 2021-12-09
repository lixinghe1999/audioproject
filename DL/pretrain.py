import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from data import NoisyCleanSet, IMUSPEECHSet, transfer_function_generator, read_transfer_function, weighted_loss
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
seg_len_mic = 2560
overlap_mic = 2240
rate_mic = 16000
seg_len_imu = 256
overlap_imu = 224
rate_imu = 1600
segment = 4
stride = 1
T = 15
N = 4
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(segment * rate_mic/(seg_len_mic-overlap_mic)) + 1
class nonlinear_function(nn.Module):
    def __init__(self, start):
        super(nonlinear_function, self).__init__()
        self.start = start
        self.t1 = torch.nn.Parameter(torch.from_numpy(self.start.astype('float32'))[None, None, :], )
        #self.t1 = torch.nn.Parameter(torch.randn(1, freq_bin_high-freq_bin_low, time_bin))
        self.n1 = nn.ReLU(inplace=True)
        self.t2 = torch.nn.Parameter(torch.randn(1, freq_bin_high - freq_bin_low, time_bin))
        self.n2 = nn.ReLU(inplace=True)
        self.t3 = torch.nn.Parameter(torch.randn(1, freq_bin_high - freq_bin_low, time_bin))
        self.n3 = nn.ReLU(inplace=True)
    def forward(self, x):
        # output1 = self.t1 * x
        # output2 = self.n2(self.t2 * output1)
        # output3 = self.n3(self.t3 * output2)
        return x
if __name__ == "__main__":

    EPOCH = 20
    BATCH_SIZE = 8
    lr = 0.2
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    transfer_function, variance, hist, bins, noise_hist, noise_bins, noise = read_transfer_function('transfer_function')
    response = np.tile(np.expand_dims(transfer_function[0, :], axis=1), (1, time_bin))
    model = nonlinear_function(response).to(device)
    Loss = nn.L1Loss()
    IMU_dataset = IMUSPEECHSet('imuexp4.json', 'wavexp4.json', minmax=(1, 1))
    train_loader = Data.DataLoader(dataset=IMU_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    with torch.no_grad():
        L = []
        for x, y in train_loader:
            x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
            predict = model(y)
            L.append(Loss(predict, x).item())
    torch.save(model.state_dict(), 'withouttrain' + '_' + str(np.mean(L)) + '.pth')
    # for epoch in range(EPOCH):
    #     for x, y in tqdm(train_loader):
    #         x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
    #         predict = model(y)
    #         loss = Loss(predict, x)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     with torch.no_grad():
    #         L = []
    #         for x, y in train_loader:
    #             x, y = x.to(device=device, dtype=torch.float), y.to(device=device,  dtype=torch.float)
    #             predict = model(y)
    #             L.append(Loss(predict, x).item())
    #     torch.save(model.state_dict(), 'transferfunction' + '_' + str(epoch) + '_' + str(np.mean(L)) + '.pth')
    #             # fig, axs = plt.subplots(4)
                # axs[0].imshow(x.cpu()[0, 0, :, :], aspect='auto')
                # axs[1].imshow(y.cpu()[0, 0, :, :], aspect='auto')
                # axs[2].imshow(predict.cpu()[0, 0, :, :], aspect='auto')
                # axs[3].imshow(synthetic(y.cpu()[0, 0, :, :], transfer_function, hist, bins, noise_hist, noise_bins), aspect='auto')
                # plt.show()