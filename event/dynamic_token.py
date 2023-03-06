import time
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.dyvit import AVnet_Dynamic
import warnings
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore")

def train_step(model, input_data, optimizer, criteria, label, mode='dynamic'):
    audio, image = input_data
    # cumulative loss
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
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False)
    model.eval()
    compress_level = []
    error = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            audio, image, text, _ = batch
            output_cache, output = model(audio.to(device), image.to(device), 'no_exit')

            gate_label = model.label(output_cache, text)
            gate_label = torch.argmax(gate_label[0], dim=-1, keepdim=True).cpu().numpy()
            if torch.argmax(output).cpu() == text:
                correct += 1
                compress_level.append(gate_label)
            elif gate_label[0] == 11 and gate_label[1] == 11:
                error += 1
            else:
                compress_level.append(gate_label)
    compress_level = np.concatenate(compress_level, axis=-1)
    compress_diff = np.abs(compress_level[0] - compress_level[1])
    compress_diff = np.bincount(compress_diff)
    compress_audio = np.bincount(compress_level[0])
    compress_image = np.bincount(compress_level[1])
    print("compression level difference:", compress_diff / len(test_loader))
    print("audio compression level:", compress_audio / len(test_loader))
    print("image compression level:", compress_image / len(test_loader))
    print("overall accuracy:", 1 - error / len(test_loader))
    print("final layer accuracy:", correct / len(test_loader))
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
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=4, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criteria = torch.nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(20):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            train_step(model, input_data=(audio.to(device), image.to(device)), optimizer=optimizer,
                           criteria=criteria, label=text.to(device), mode=mode)
        scheduler.step()
        model.eval()
        acc = [0] * 24; count = [0] * 24
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a, e, _ = test_step(model, input_data=(audio.to(device), image.to(device)), label=text, mode=mode)
                acc[e-1] += a
                count[e-1] += 1
        mean_acc = []
        for i in range(len(acc)):
            if count[i] == 0:
                mean_acc.append(0)
            else:
                mean_acc.append(acc[i]/count[i])
        print('epoch', epoch)
        print('accuracy for early-exits:', mean_acc)
        if np.mean(mean_acc) > best_acc:
            best_acc = np.mean(mean_acc)
            torch.save(model.state_dict(), str(args.task) + '_' + str(epoch) + '_' + str(mean_acc[-1]) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='train')
    parser.add_argument('-m', '--mode', default='dynamic')
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=4, type=int)
    args = parser.parse_args()
    mode = args.mode
    workers = args.worker
    batch_size = args.batch
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    model = AVnet_Dynamic().to(device)
    model.image.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'], strict=False)

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

