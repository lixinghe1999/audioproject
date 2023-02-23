import time
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.gate_model import AVnet_Gate
import warnings
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore")

def train_step(model, input_data, optimizers, criteria, label, mode='dynamic'):
    audio, image = input_data
    # cumulative loss
    optimizer = optimizers[0]
    output_cache, output = model(audio, image, mode)
    optimizer.zero_grad()
    loss = criteria(output, label)
    loss.backward()
    optimizer.step()
    return loss.item()
def test_step(model, input_data, label, mode='dynamic'):
    audio, image = input_data
    t_start = time.time()
    output_cache, output = model(audio, image, mode)
    l = time.time() - t_start
    acc = (torch.argmax(output, dim=-1).cpu() == label).sum()/len(label)
    return acc.item(), len(output_cache['audio']) + len(output_cache['image']), l
def profile(model, test_dataset):
    ckpt_name = '28_0.5.pth'
    model.load_state_dict(torch.load(ckpt_name))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False)
    model.eval()
    compress_level = []
    error = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            audio, image, text, _ = batch
            output_cache, output = model(audio.to(device), image.to(device), (-1, -1, -1))

            gate_label = model.label(output_cache, text)
            gate_label = torch.argmax(gate_label, dim=-1, keepdim=True).cpu().numpy()
            if torch.argmax(output, dim=-1).cpu() != text:
                error += 1
            compress_level.append(gate_label)
    compress_level = np.concatenate(compress_level, axis=-1)
    compress_diff = np.abs(compress_level[0] - compress_level[1])
    compress_diff = np.bincount(compress_diff)
    compress_audio = np.bincount(compress_level[0])
    compress_image = np.bincount(compress_level[1])
    print("compression level difference:", compress_diff / len(test_loader))
    print("compression level distribution:", compress_audio / len(test_loader), compress_image / len(test_loader))
    print("overall accuracy:", 1 - error / len(test_loader))

def test(model, test_dataset):
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False)
    model.eval()
    acc = [[0], [], [], [], [], [], [], []]
    with torch.no_grad():
        for batch in tqdm(test_loader):
            audio, image, text, _ = batch
            a, e, _ = test_step(model, input_data=(audio.to(device), image.to(device)), label=text, mode='dynamic')
            acc[e - 1] += [a]
    mean_acc = []
    for ac in acc:
        mean_acc.append(np.mean(ac))
    print('accuracy for early-exits:', mean_acc)
def update_lr(optimizer, multiplier = .1):
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = param_group['lr'] * multiplier
    optimizer.load_state_dict(state_dict)
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=64, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=64, shuffle=False)
    for param in model.audio.parameters():
        param.requires_grad = False
    for param in model.image.parameters():
        param.requires_grad = False
    optimizers = [torch.optim.Adam(model.fusion_parameter(), lr=.0001, weight_decay=1e-4)]
    criteria = torch.nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(20):
        mode = 'dynamic'
        model.train()
        if epoch % 4 == 0 and epoch > 0:
            for optimizer in optimizers:
                update_lr(optimizer, multiplier=.4)
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            train_step(model, input_data=(audio.to(device), image.to(device)), optimizers=optimizers,
                           criteria=criteria, label=text.to(device), mode=mode)
        model.eval()
        acc = [[0], [], [], [], [], [], [], []]
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a, e, _ = test_step(model, input_data=(audio.to(device), image.to(device)), label=text, mode=mode)
                acc[e-1] += [a]
        mean_acc = []
        for ac in acc:
            mean_acc.append(np.mean(ac))
        print('epoch', epoch)
        print('accuracy for early-exits:', mean_acc)
        if np.mean(mean_acc) > best_acc:
            best_acc = np.mean(mean_acc)
            torch.save(model.state_dict(), str(epoch) + '_' + str(mean_acc[-1]) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    model = AVnet_Gate().to(device)
    model.audio.load_state_dict(torch.load('A_4_0.5673682.pth'))
    model.image.load_state_dict(torch.load('V_6_0.5151336.pth'))
    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    if args.task == 'train':
        train(model, train_dataset, test_dataset)
    elif args.task == 'test':
        test(model, test_dataset)
    elif args.task == 'profile':
        profile(model, test_dataset)

