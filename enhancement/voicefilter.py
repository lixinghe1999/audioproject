import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import time

class voicefilter(nn.Module):
    def __init__(self, num_freq=201, lstm_dim=400, emb_dim=256, fc1_dim=600):
        super(voicefilter, self).__init__()
        self.conv = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn2
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)), # (9, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn5
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)), # (17, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn6
            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)), # (33, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn7
            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)), # (65, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            8 * num_freq + emb_dim,
            lstm_dim,
            batch_first=True,
            bidirectional=True)

        self.fc1 = nn.Linear(2*lstm_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, num_freq)

    def forward(self, x, dvec):
        # x: [B, T, num_freq]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_freq]
        x = self.conv(x)
        # x: [B, 8, T, num_freq]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]

        # dvec: [B, emb_dim]
        dvec = dvec.unsqueeze(1)
        print(x.shape, dvec.shape)
        dvec = dvec.repeat(1, x.size(1), 1)
        # dvec: [B, T, emb_dim]

        x = torch.cat((x, dvec), dim=2) # [B, T, 8*num_freq + emb_dim]

        x, _ = self.lstm(x) # [B, T, 2*lstm_dim]
        x = F.relu(x)
        x = self.fc1(x) # x: [B, T, fc1_dim]
        x = F.relu(x)
        x = self.fc2(x) # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        return x
class voicefilter_lite(nn.Module):
    def __init__(self, num_freq=601, lstm_dim=512, emb_dim=256, fc1_dim=600, fc2_dim=601):
        super(voicefilter_lite, self).__init__()
        self.lstm = nn.LSTM(
            1*num_freq + emb_dim,
            lstm_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=False)

        self.fc1 = nn.Linear(lstm_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

    def forward(self, x, dvec):
        dvec = dvec.unsqueeze(1)
        dvec = dvec.repeat(1, x.size(1), 1)

        x = torch.cat((x, dvec), dim=2) # [B, T, 8*num_freq + emb_dim]
        x, _ = self.lstm(x) # [B, T, 2*lstm_dim]
        x = F.relu(x)
        x = self.fc1(x) # x: [B, T, fc1_dim]
        x = F.relu(x)
        x = self.fc2(x) # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        return x


class LinearNorm(nn.Module):
    def __init__(self, lstm_hidden, emb_dim):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(lstm_hidden, emb_dim)

    def forward(self, x):
        return self.linear_layer(x)


class SpeechEmbedder(nn.Module):
    def __init__(self, num_mels=40, n_fft=512, emb_dim=256, lstm_hidden=768, lstm_layers=3, window=80, stride=40):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(num_mels,
                            lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.proj = LinearNorm(lstm_hidden, emb_dim)
        self.mel_basis = librosa.filters.mel(sr=16000,
                                             n_fft=n_fft,
                                             n_mels=num_mels)
        self.window = window
        self.stride = stride
        self.num_mels = num_mels
        self.lstm_hidden = lstm_hidden
        self.emb_dim = emb_dim

    def get_mel(self, y):
        y = librosa.core.stft(y=y, n_fft=512, hop_length=160, win_length=400, window='hann')
        magnitudes = np.abs(y) ** 2
        mel = np.dot(self.mel_basis, magnitudes)
        mel = np.log10(mel + 1e-6)
        return mel.transpose((1, 0, 2))

    def forward(self, clean, device='cuda'):
        batch = clean.shape[0]
        mel = self.get_mel(clean)
        mel = torch.from_numpy(mel).float().to(device)
        # (num_mels, T)
        mels = mel.unfold(2, self.window, self.stride) # (num_mels, T', window)
        mels = mels.permute(0, 2, 3, 1) # (T', window, num_mels)
        mels = mels.reshape(-1, self.window, self.num_mels)
        x, _ = self.lstm(mels) # (T', window, lstm_hidden)
        x = x[..., -1, :] # (T', lstm_hidden), use last frame only
        x = self.proj(x) # (T', emb_dim)
        x = x.reshape(batch, -1, self.emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True) # (T', emb_dim)
        x = x.sum(1) / x.size(1) # (emb_dim), average pooling over time frames
        return x
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
    step = 100
    with torch.no_grad():
        for i in range(step):
            model(*input)
    return (time.time() - t_start)/step
if __name__ == "__main__":
    # ckpt = torch.load('embedder.pt')
    # embedder.load_state_dict(ckpt)
    # audio = np.random.rand(2, 48000)
    # embedder = SpeechEmbedder().to('cuda')
    # vector = embedder(audio)
    # print(vector.shape)

    device = 'cuda'
    vector = torch.rand(2, 256).to(device)
    noisy = torch.rand(2, 300, 601).to(device)
    model = voicefilter(num_freq=601).to(device)

    ckpt = torch.load('voiceSplit-trained-with-MSE-GE2E-Seungwonpark-best_checkpoint.pt')
    model.load_state_dict(ckpt['model'])
    # clean = model(noisy, vector)
    #
    # print(clean.shape)
    # print(model_size(model))
    # print(model_speed(model, [noisy, vector]))
