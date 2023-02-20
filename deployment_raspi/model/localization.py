import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
class Encoder(nn.Module):
    def __init__(self, n_freq=512):
        super(Encoder, self).__init__()
        self.GRU1 = nn.GRU(n_freq, 200, batch_first=True)
        self.relu1 = nn.ReLU()
        self.GRU2 = nn.GRU(200, 100, batch_first=True)
        self.relu2 = nn.ReLU()
        self.z_mean = nn.Linear(100, 100)
        self.z_variance = nn.Linear(100, 100)
    def reparameterize(self, m, var):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps * std + m

    def forward(self, input):
        z = self.GRU1(input)[0]
        z = self.relu1(z)
        z = self.GRU2(z)[1]
        z = self.relu2(z)
        m = self.z_mean(z)
        var = self.z_variance(z)
        z = self.reparameterize(m, var)
        return [z, input, m, var]
class SubNet(nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()
        self.Fc = nn.Linear(200, 100)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

        self.sound1 = nn.Linear(100, 50)
        self.sound2 = nn.Linear(50, 10)
        self.sound3 = nn.Linear(10, 1)

        self.aoa1 = nn.Linear(100, 50)
        self.aoa2 = nn.Linear(50, 10)
        self.aoa3 = nn.Linear(10, 1)

        self.dis1 = nn.Linear(100, 50)
        self.dis2 = nn.Linear(50, 10)
        self.dis3 = nn.Linear(10, 5)

    def forward(self, input):
        features = self.drop(self.relu(self.Fc(input)))

        sound = F.relu(self.sound1(features))
        sound = F.relu(self.sound2(sound))
        sound = torch.sigmoid(self.sound3(sound))

        aoa = F.relu(self.aoa1(features))
        aoa = F.relu(self.aoa2(aoa))
        aoa = torch.sigmoid(self.aoa3(aoa))

        dis = F.relu(self.dis1(features))
        dis = F.relu(self.dis2(dis))
        dis = F.softmax(self.dis3(dis), dim=-1)
        return [sound, aoa, dis]


class DeepEar(nn.Module):
    def __init__(self):
        super(DeepEar, self).__init__()
        self.left = Encoder()
        self.right = Encoder()
        self.decode1 = nn.Linear(400, 512)
        self.decode2 = nn.Linear(512, 400)
        self.decode3 = nn.Linear(400, 200)
        self.subnet = SubNet()
    def forward(self, Binaural):
        Binaural = Binaural.permute((0, 1, 3, 2))
        # TOFIX: add cross-correlation based on GCC-PHAT
        left = Binaural[:, 0, ...]
        right = Binaural[:, 1, ...]
        left, input_left, m_left, var_left = self.left(left)
        right, input_right, m_right, var_right = self.right(right)
        features = torch.cat([left, right, left - right, right - left], dim=1)
        features = features.reshape((-1, 400))
        features = self.decode1(features)
        features = self.decode2(features)
        features = self.decode3(features)
        output = self.subnet(features)
        return output

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepEar().to(device)
    Binaural = torch.rand((1, 2, 300, 512)).to(device)
    cross = torch.rand((1, 1, 100)).to(device)
    # left = Binaural[:, 0, ...]
    # right = Binaural[:, 1, ...]
    output = model(Binaural, cross)
