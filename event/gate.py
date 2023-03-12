import time
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.gate_model import AVnet_Gate, Gate
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
    acc = (torch.argmax(output, dim=-1).cpu() == label).sum() / len(label)
    return acc.item(), len(output_cache['audio']) + len(output_cache['image']), l
def gate_train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=1, shuffle=False)
    gate = Gate(option=1).to(device)
    model.eval()
    model.gate = gate
    for param in model.audio.parameters():
        param.requires_grad = False
    for param in model.image.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.get_parameters(), lr=.00001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    best_acc = 0
    for epoch in range(5):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            optimizer.zero_grad()
            [compress, acc] = model.gate_train(audio.to(device), image.to(device), text.to(device))
            if idx % 200 == 0 and idx > 0:
                print(compress, acc)
            #     writer.add_scalars('Train/compression', {'audio': compress[0], 'image': compress[1]}, idx + epoch * len(train_loader))
            #     writer.add_scalar('Train/acc', acc, idx + epoch * len(train_loader))
            optimizer.step()
        scheduler.step()
        model.eval()
        acc = [0] * 24; count = [0] * 24
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a, e, _ = test_step(model, input_data=(audio.to(device), image.to(device)), label=text, mode='gate')
                acc[e-1] += a
                count[e-1] += 1
        mean_acc = []
        for i in range(len(acc)):
            if count[i] == 0:
                mean_acc.append(0)
            else:
                mean_acc.append(acc[i]/count[i])
        acc_all = np.round(mean_acc, 3)
        acc_avg = np.round(np.sum(acc) / np.sum(count), 3)
        comp_all = np.round(np.array(count) / np.sum(count), 3)
        comp_avg = np.mean(comp_all * np.linspace(1, 24, 24))
        print('epoch', epoch, 'trained gate exit')
        print('accuracy for early-exits:', acc_all.tolist())
        print('mean accuracy ', acc_avg)
        print('compression level distribution:', comp_all.tolist())
        print('mean compression level:', comp_avg )
        if acc_avg > best_acc:
            best_acc = acc_avg
            torch.save(model.state_dict(), 'gate_' + str(args.task) + '_' + str(epoch) + '_' +
                       str(acc_avg) + '.pth')
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
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=1, shuffle=False)
    for param in model.audio.parameters():
        param.requires_grad = False
    for param in model.image.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.get_parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criteria = torch.nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(10):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            train_step(model, input_data=(audio.to(device), image.to(device)), optimizer=optimizer,
                           criteria=criteria, label=text.to(device), mode='dynamic')
        scheduler.step()
        model.eval()
        acc = [0] * 24; count = [0] * 24
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a, e, _ = test_step(model, input_data=(audio.to(device), image.to(device)), label=text, mode='dynamic')
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
            torch.save(model.state_dict(), 'gate_' + str(args.task) + '_' + str(epoch) + '_' + str(mean_acc[-1]) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='train')
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=4, type=int)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    model = AVnet_Gate().to(device)
    model.audio.load_state_dict(torch.load('vanilla_A_6_0.5303089942924621.pth'))
    model.image.load_state_dict(torch.load('vanilla_V_7_0.5041330446762449.pth'))

    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))

    if args.task == 'train':
        train(model, train_dataset, test_dataset)
    elif args.task == 'test':
        test(model, test_dataset)
    elif args.task == 'gate_train':
        model.load_state_dict(torch.load('gate_train_9_0.6756756756756757.pth'))
        gate_train(model, train_dataset, test_dataset)
    elif args.task == 'profile':
        profile(model, test_dataset)

