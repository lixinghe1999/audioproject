import torch
import torch.utils.data as Data
from data import NoisyCleanSet
from unet import UNet
import matplotlib.pyplot as plt
import numpy as np
def tensor2image(t):
    t = t.squeeze().numpy()
    t = t[0] + t[1] * 1j
    return np.abs(t)
if __name__ == "__main__":
    fig, axs = plt.subplots(3, 1)
    EPOCH = 10
    BATCH_SIZE = 1
    sample_rate = 16000
    segment = 4
    stride = 1
    pad = True
    model = UNet(2, 2)
    model.load_state_dict(torch.load('checkpoint.pth'))
    dataset_train = NoisyCleanSet('dataset', length=segment, stride=stride, pad=pad, sample_rate=sample_rate)
    loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            predict = model(x)
            axs[0].imshow(tensor2image(x), aspect='auto')
            axs[1].imshow(tensor2image(y), aspect='auto')
            axs[2].imshow(tensor2image(predict), aspect='auto')
            plt.show()
            break