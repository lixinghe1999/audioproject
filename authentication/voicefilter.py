import torch
import torch.nn as nn
import librosa
import numpy as np


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
if __name__ == "__main__":
    audio = np.random.rand(2, 48000)
    embedder = SpeechEmbedder().to('cuda')
    vector = embedder(audio)
    print(vector.shape)
    ckpt = torch.load('embedder.pt')
    embedder.load_state_dict(ckpt)
