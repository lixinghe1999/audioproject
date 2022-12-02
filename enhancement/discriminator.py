import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
import torch.nn as nn
import torch

def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score

class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class Discriminator_spectrogram(nn.Module):
    def __init__(self, ndf=16, in_channel=2):
        super(Discriminator_spectrogram, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.PReLU(2*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.PReLU(8*ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            LearnableSigmoid(1)
        )
    def batch_pesq(self, clean, noisy):

        pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
        pesq_score = np.array(pesq_score)
        if -1 in pesq_score:
            return None
        pesq_score = (pesq_score - 1) / 3.5
        return torch.FloatTensor(pesq_score).to('cuda')

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)

class Discriminator_time(nn.Module):
    def __init__(self):
        super(Discriminator_time, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)),
        ])

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=1)
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [Discriminator_time() for _ in range(3)]
        )

        self.pooling = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)]
        )

    def forward(self, x):
        ret = list()
        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))
        return ret
class SingleScaleDiscriminator(nn.Module):
    def __init__(self):
        super(SingleScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [Discriminator_time() for _ in range(1)]
        )

    def forward(self, x):
        return self.discriminators(x)

if __name__ == '__main__':
    model_s = Discriminator_spectrogram()
    model_t = Discriminator_time()
    model_m = MultiScaleDiscriminator()

    x_t = torch.randn(4, 1, 80000)
    x_s = torch.randn(4, 1, 256, 200)

    #features, score = model_t(x_t)
    multi_scale_output = model_m(x_t)
    for scale in multi_scale_output:
        feats, score = scale
        for feat in feats:
            print(feat.shape)
        print('score', score.shape)

    # metric = model_s(x_s, x_s)
    # print(metric.shape)
