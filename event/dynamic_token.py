import time
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.dyvit import AVnet_Dynamic
from utils.losses import DistillDiffPruningLoss_dynamic
import warnings
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore")

def train_step(model, input_data, optimizer, criteria, label, mode='dynamic'):
    # cumulative loss
    outputs = model(**input_data)
    optimizer.zero_grad()
    loss = criteria(input_data, outputs, label)
    loss.backward()
    optimizer.step()
    return loss.item()
def test_step(model, input_data, label, mode='dynamic'):
    audio, image = input_data
    output = model(audio, image)
    acc = (torch.argmax(output, dim=-1).cpu() == label).sum()/len(label)
    return acc.item()
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
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    best_acc = 0
    # model.eval()
    # acc = []
    # with torch.no_grad():
    #     for batch in tqdm(test_loader):
    #         audio, image, text, _ = batch
    #         a = test_step(model, input_data=(audio.to(device), image.to(device)), label=text, mode=mode)
    #         acc.append(a)
    #     mean_acc = np.mean(acc)
    #     print('accuracy:', mean_acc)

    for epoch in range(20):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            train_step(model, input_data=(audio.to(device), image.to(device)), optimizer=optimizer,
                           criteria=criteria, label=text.to(device), mode=mode)
        scheduler.step()
        model.eval()
        acc = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a = test_step(model, input_data=(audio.to(device), image.to(device)), label=text, mode=mode)
                acc.append(a)
        mean_acc = np.mean(acc)
        print('epoch', epoch)
        print('accuracy:', mean_acc)
        if mean_acc > best_acc:
            best_acc = mean_acc
            torch.save(model.state_dict(), str(args.task) + '_' + str(epoch) + '_' + str(mean_acc) + '.pth')
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
    pruning_loc = [3, 6, 9]
    base_rate = 0.7
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    model = AVnet_Dynamic(pruning_loc=pruning_loc, token_ratio=token_ratio, pretrained=False, distill=True).to(device)
    model.load_state_dict(torch.load('train_6_0.6778193269041527.pth'), strict=False)


    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))

    if args.task == 'train':
        train(model, train_dataset, test_dataset)
    elif args.task == 'distill':
        teacher_model = AVnet_Dynamic(pruning_loc=(), pretrained=False).to(device)
        teacher_model.load_state_dict(torch.load('train_6_0.6778193269041527.pth'), strict=False)
        teacher_model.eval()
        criteria = DistillDiffPruningLoss_dynamic(teacher_model, torch.nn.CrossEntropyLoss(), clf_weight=1.0, keep_ratio=token_ratio,
                                                  mse_token=True, ratio_weight=2.0, distill_weight=0.5)
        train(model, train_dataset, test_dataset)
    elif args.task == 'test':
        test(model, test_dataset)
    elif args.task == 'profile':
        profile(model, test_dataset)

