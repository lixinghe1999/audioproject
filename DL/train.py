import torch
import torch.utils.data as Data
import torch.nn as nn
from data import NoisyCleanSet
from unet import UNet

if __name__ == "__main__":
    torch.manual_seed(1)
    EPOCH = 30
    BATCH_SIZE = 16
    sample_rate = 16000
    segment = 2
    stride = 1
    lr = 0.001
    pad = True
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model = UNet(1, 1).to(device)

    dataset_train = NoisyCleanSet('dataset', length=segment, stride=stride, pad=pad, sample_rate=sample_rate)
    loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    loss_func = nn.MSELoss()
    for epoch in range(EPOCH):
        loss_train = 0
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            predict = model(x).to(device)
            loss = loss_func(predict, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        print('Epoch :', epoch, '|', 'train_loss:%.6f' % (loss_train/(step+1)))
        torch.save(model.state_dict(), 'checkpoint.pth')
    print('________________________________________')
    print('finish training')