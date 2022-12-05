from fullsubnet import FullSubNet
from vibvoice import A2net
from dataset import NoisyCleanSet
from model_zoo import test_vibvoice, test_fullsubnet
import torch

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
vibvoice = A2net(inference=False).to(device)
fullsubnet = FullSubNet(num_freqs=256, num_groups_in_drop_band=1).to(device)

ckpt_vibvoice = torch.load()
vibvoice.load_state_dict(ckpt_vibvoice)

ckpt_fullsubnet = torch.load()
fullsubnet.load_state_dict(ckpt_fullsubnet)

for noise in ['background.json', 'dev.json', 'music.json']:
    dataset = NoisyCleanSet(['json/train_gt.json', 'json/' + noise, 'json/train_imu.json'],
                            person=['he'], time_domain=False, simulation=True, ratio=-0.2)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=1, shuffle=False)
    with torch.no_grad():
        for data in test_loader:
            acc, noise, clean = data
            predict1, gt = test_vibvoice(vibvoice, acc, noise, clean, device, data=True)
            predict2, gt = test_fullsubnet(fullsubnet, acc, noise, clean, device, data=True)
