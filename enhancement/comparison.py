from fullsubnet import FullSubNet
from vibvoice import A2net
from dataset import NoisyCleanSet
from model_zoo import test_vibvoice, test_fullsubnet
import soundfile as sf
import torch
'''
This Script will save the result from vibvoice and fullsubnet for demonstration 
'''

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
vibvoice = A2net(inference=False).to(device)
fullsubnet = FullSubNet(num_freqs=256, num_groups_in_drop_band=1).to(device)
fullsubnet = torch.nn.DataParallel(fullsubnet, device_ids=[0])

ckpt_vibvoice = torch.load('vibvoice.pth')
vibvoice.load_state_dict(ckpt_vibvoice)

ckpt_fullsubnet = torch.load('fullsubnet.pth')
fullsubnet.load_state_dict(ckpt_fullsubnet)
if __name__ == "__main__":
    count = 0
    noises = ['background.json', 'dev.json', 'music.json']
    noise = 'background.json'
    dataset = NoisyCleanSet(['json/train_gt.json', 'json/' + noise, 'json/train_imu.json'],
                            person=['he'], time_domain=False, simulation=True, ratio=-0.2)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)
    print(len(dataset))
    with torch.no_grad():
        for data in test_loader:
            acc, noise, clean = data
            [pesq1, snr1, lsd1], predict1, gt = test_vibvoice(vibvoice, acc, noise, clean, device, data=True)
            [pesq2, snr2, lsd2], predict2, gt = test_fullsubnet(fullsubnet, acc, noise, clean, device, data=True)
            if (pesq1 - pesq2) > 0.4:
                sf.write(str(count) + '_vibvoice.wav', predict1, 16000)
                sf.write(str(count) + '_fullsubnet.wav', predict2, 16000)
                count += 1

