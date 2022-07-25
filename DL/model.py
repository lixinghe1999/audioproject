import time

import torch.nn as nn
import torch
import torch.nn.functional as F
def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2

    return size_all_mb
def model_speed(model, input):
    t_start = time.time()
    step = 200
    with torch.no_grad():
        for i in range(step):
            model(*input)
    return (time.time() - t_start)/step
class attention_block(nn.Module):
    def __init__(self, channels, freq):
        super(attention_block, self).__init__()
        self.freq = freq
        self.channels = channels
        self.conv1 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=(1, 1))
        self.conv2 = nn.Conv1d(self.channels[1] * self.freq, self.channels[0], kernel_size=9, padding=4)
        self.conv3 = nn.Conv2d(2 * self.channels[0], self.channels[0], kernel_size=(1, 1))
    def forward(self, x):
        b, c, f, t = x.shape
        x1 = self.conv1(x)
        x1 = x1.reshape((b, -1, t))
        x1 = self.conv2(x1)
        x1 = x1[:, :, None, :]
        x1 = x1 * x
        x1 = torch.cat([x1, x], dim=1)
        x1 = self.conv3(x1)
        return x1

class Feature_embeddings(nn.Module):
    def __init__(self, channels, kernels, paddings, max_poolings, attentions):
        super(Feature_embeddings, self).__init__()
        layers = []
        for i in range(1, len(channels)):
            layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=kernels[i-1], padding=paddings[i-1]))
            layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
            if i in attentions:
                layers.append(attention_block([channels[i], int(channels[i]/2)], attentions[i]))
            if i in max_poolings:
                layers.append(nn.MaxPool2d(kernel_size=max_poolings[i]))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
class Fusion(nn.Module):
    def __init__(self, LSTM_channels, attention_channels, freq, fc_layers):
        super(Fusion, self).__init__()
        self.attention = attention_block([attention_channels, int(attention_channels/2)], freq)
        self.biLSTM = nn.LSTM(LSTM_channels[0], LSTM_channels[1], 2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * LSTM_channels[1], fc_layers[0])
        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.fc3 = nn.Linear(fc_layers[1], fc_layers[2])
    def forward(self, x):
        b, c, f, t = x.shape
        x = x.permute((0, 3, 1, 2))
        x = x.reshape(b, t, -1)
        self.biLSTM.flatten_parameters()
        x, _ = self.biLSTM(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x).permute(0, 2, 1)
        return x

class A2net(nn.Module):
    def __init__(self):
        super(A2net, self).__init__()
        self.IMU_branch = Feature_embeddings(channels=[1, 16, 64, 128, 8], kernels=[(3, 3), (3, 3), (3, 3), (1, 1)],
                                       paddings=[(1, 1), (1, 1), (1, 1), (0, 0)], max_poolings={1: (3, 1)}, attentions={2:11})

        self.Audio_branch = Feature_embeddings(channels=[1, 16, 64, 128, 128, 8], kernels=[(5, 5), (3, 3), (3, 3), (3, 3), (1, 1)],
                                             paddings=[(2, 2), (1, 1), (1, 1), (1, 1), (0, 0)], max_poolings={1: (3, 1), 3: (2, 1)},
                                             attentions={2:88, 4:44})
        self.fusion = Fusion(LSTM_channels=[440, 300], attention_channels=8, freq=55, fc_layers=[300, 300, 264])
    def forward(self, x1, x2):
        x1 = self.IMU_branch(x1)
        x2 = self.Audio_branch(x2)
        x = torch.cat([x1, x2], dim=2)
        # [B, C=8, F=44+11, T]
        x = self.fusion(x)
        x = x[:, None, :, :]
        return x

if __name__ == "__main__":

    imu = torch.rand(1, 1, 33, 151)
    audio = torch.rand(1, 1, 264, 151)
    model = A2net()
    size_all_mb = model_size(model)
    print('model size: {:.3f}MB'.format(size_all_mb))

    latency = model_speed(model, [imu, audio])
    print('model latency: {:.3f}S'.format(latency))


