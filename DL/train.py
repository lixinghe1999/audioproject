import torch
import torch.utils.data as Data
import torch.nn as nn
from data import NoisyCleanSet
from unet import UNet
from tqdm import tqdm
import argparse
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    args = parser.parse_args()
    return args
if __name__ == "__main__":

    device_ids = [0]
    EPOCH = 50
    BATCH_SIZE = 64
    sample_rate = 16000
    segment = 3
    stride = 1
    lr = 0.001 / len(device_ids)
    pad = True
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model = UNet(1, 1)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    train_dataset1 = NoisyCleanSet('speech100.json')
    train_dataset2 = NoisyCleanSet('speech360.json')
    #train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
    train_loader = Data.DataLoader(dataset=train_dataset1, batch_size=BATCH_SIZE * len(device_ids), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    loss_func = nn.MSELoss()
    for epoch in range(EPOCH):
        for x, y in tqdm(train_loader):
            x, y = x.to(device=device_ids[0], dtype=torch.float), y.to(device = device_ids[0])
            predict = model(x)
            loss = loss_func(predict, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f'checkpoint{epoch}.pth')
    print('________________________________________')
    print('finish training')