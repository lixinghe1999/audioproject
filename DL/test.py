import torch
import torch.utils.data as Data
from data import NoisyCleanSet, IMUSPEECHSet
from unet import UNet
import matplotlib.pyplot as plt
import numpy as np
def tensor2image(t):
    t = t.cpu().squeeze().numpy()
    #t = t[0] + t[1] * 1j
    return np.abs(t)
if __name__ == "__main__":

    EPOCH = 10
    BATCH_SIZE = 1
    model = UNet(1, 1)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load("checkpoint31.pth"))
    train_dataset1 = NoisyCleanSet('speech100.json')
    train_dataset2 = NoisyCleanSet('speech360.json')
    IMU_dataset = IMUSPEECHSet('imuexp4.json', 'wavexp4.json')
    # train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
    train_loader = Data.DataLoader(dataset=train_dataset2, batch_size=BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device=0, dtype=torch.float), y.to(device=0)
            #x = x.to(device=0, dtype=torch.float)
            predict = model(x)
            fig, axs = plt.subplots(3)
            axs[0].imshow(tensor2image(x), aspect='auto')
            axs[1].imshow(tensor2image(y), aspect='auto')
            axs[2].imshow(tensor2image(predict), aspect='auto')
            plt.show()