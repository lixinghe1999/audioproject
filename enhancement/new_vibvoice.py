import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

freq_bin_high = 33
def synthetic(clean, transfer_function, N):
    time_bin = clean.shape[-1]
    index = np.random.randint(0, N)
    f = transfer_function[index, 0]
    f = f / np.max(f)
    v = transfer_function[index, 1] / np.max(f)
    response = np.tile(np.expand_dims(f, axis=1), (1, time_bin))
    for j in range(time_bin):
        response[:, j] += np.random.normal(0, v, (freq_bin_high))
    response = torch.from_numpy(response).to(clean.device)
    acc = clean[..., :freq_bin_high, :] * response
    return acc

class vibvoice(nn.Module):
    def __init__(self, num_freq=321, lstm_dim=400, emb_dim=33, fc1_dim=600, fc2_dim=321):
        super(vibvoice, self).__init__()
        self.conv = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(1, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn2
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((4, 4, 2, 2)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 2)), # (9, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn5
            nn.ZeroPad2d((8, 8, 2, 2)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 4)), # (17, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn6
            nn.ZeroPad2d((16, 16, 2, 2)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 8)), # (33, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn7
            nn.ZeroPad2d((32, 32, 2, 2)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 16)), # (65, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.conv_acc = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(1, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn2
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((4, 4, 2, 2)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 2)),  # (9, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn5
            nn.ZeroPad2d((8, 8, 2, 2)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 4)),  # (17, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn6
            nn.ZeroPad2d((16, 16, 2, 2)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 8)),  # (33, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn7
            nn.ZeroPad2d((32, 32, 2, 2)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 16)),  # (65, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            8 * num_freq + 8 * emb_dim,
            lstm_dim,
            batch_first=True,
            bidirectional=True)

        self.fc1 = nn.Linear(2*lstm_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

        self.transfer_function = np.load('transfer_function_EMSB_filter.npy')
        self.length_transfer_function = self.transfer_function.shape[0]
    def norm(self, x):
        mu = torch.mean(x, dim=list(range(1, x.dim())), keepdim=True)
        normed = x / (mu + 1e-5)
        return normed
    def forward(self, x, acc=None):
        # Preprocessing
        if acc == None:
            acc = synthetic(x, self.transfer_function, self.length_transfer_function)
        else:
            batch = acc.shape[0]
            acc = acc.to(x.device).reshape(batch*3, -1)
            acc = torch.abs(torch.stft(acc, 64, 32, 64, window=torch.hann_window(64, device=x.device), return_complex=True))
            acc = torch.norm(acc.reshape(batch, 3, 33, -1), p=2, dim=1)
        acc = self.norm(acc)
        x = self.norm(x)

        # x: [B, num_freq, T]
        x = x.unsqueeze(1)
        # x: [B, 1, num_freq, T]
        x = self.conv(x)
        # x: [B, 8, num_freq, T]
        x = x.permute(0, 3, 1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]

        # x: [B, num_freq, T]
        acc = acc.unsqueeze(1)
        # x: [B, 1, num_freq, T]
        acc = self.conv_acc(acc)
        # x: [B, 8, num_freq, T]
        acc = acc.permute(0, 3, 1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        acc = acc.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]

        x = torch.cat((x, acc), dim=2) # [B, T, 8*num_freq + emb_dim]

        x, _ = self.lstm(x) # [B, T, 2*lstm_dim]
        x = F.relu(x)
        x = self.fc1(x) # x: [B, T, fc1_dim]
        x = F.relu(x)
        x = self.fc2(x) # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        return x.permute(0, 2, 1)
def model_speed(model, input):
    t_start = time.time()
    step = 100
    with torch.no_grad():
        for i in range(step):
            model(*input)
    return (time.time() - t_start)/step
def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb
if __name__ == "__main__":

    device = 'cpu'
    acc = torch.rand(1, 33, 151).to(device)
    noisy = torch.rand(1, 321, 151).to(device)
    filter = vibvoice().to(device)
    clean = filter(noisy)
    print(clean.shape)
    print(model_speed(filter, [noisy]))
    print(model_size(filter))