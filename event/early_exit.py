import time
import pandas as pd
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.exit_model import AVnet_Exit
import warnings
from tqdm import tqdm
from datetime import date
import argparse

warnings.filterwarnings("ignore")
# remove annoying librosa warning
def profile(model, test_dataset):
    ckpt_name = '9_0.5913482705752054.pth'
    model.load_state_dict(torch.load(ckpt_name))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False)
    model.eval()
    model.exit = True
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    writer = pd.ExcelWriter('result_recording.xlsx', engine='xlsxwriter')
    data_frame = {'threshold': thresholds, 'latency': [], 'accuracy': [], 'accuracy_exit': [], 'exit_percentage': []}
    with torch.no_grad():
        for threshold in thresholds:
            acc = []
            ee = []
            latency = 0
            model.threshold = threshold
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a, e, l = test_step(model, input_data=(audio.to(device), image.to(device)), label=text)
                acc.append(a)
                ee.append(e)
                latency += l
            acc = np.stack(acc)
            ee = np.array(ee)
            print('threshold', threshold)
            print('latency:', latency / len(test_loader))
            print('accuracy for each exit:', np.mean(acc, axis=0, where=acc >= 0))
            # print('accuracy for early-exits:', np.mean(acc[np.arange(len(acc)), ee - 1], axis=0))
            print('early-exit percentage:', np.bincount(ee-1) / ee.shape[0])
            data_frame['latency'] += [latency / len(test_loader)]
            data_frame['accuracy'] += [np.mean(acc, axis=0, where=acc >= 0)]
            # data_frame['accuracy_exit'] += [np.mean(acc[np.arange(len(acc)), ee - 1], axis=0)]
            data_frame['exit_percentage'] += [np.bincount(ee-1) / ee.shape[0]]
    df = pd.DataFrame(data=data_frame)
    df.to_excel(writer, sheet_name=date.today().strftime("%d/%m/%Y %H:%M:%S"))
def train_step(model, input_data, optimizer, criteria, label):
    audio, image = input_data
    outputs = model(audio, image)
    optimizer.zero_grad()
    loss = 0
    for i, output in enumerate(outputs):
        loss += (i+1)/12 * criteria(output, label)
    loss.backward()
    optimizer.step()
    return loss.item()
def test_step(model, input_data, label):
    audio, image = input_data
    outputs = model(audio, image)


    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    early_acc = np.zeros((len(thresholds)))
    early_exit = np.zeros((len(thresholds)))
    for j, thres in enumerate(thresholds):
        for i, output in enumerate(outputs):
            max_confidence = torch.max(torch.softmax(output, dim=-1))
            if max_confidence > thres:
                acc = (torch.argmax(output, dim=-1).cpu() == label).sum() / len(label)
                early_acc[j] = acc
                early_exit[j] = (i+1)/12
                break
            else:
                acc = (torch.argmax(output, dim=-1).cpu() == label).sum() / len(label)
                early_acc[j] = acc
                early_exit[j] = (i + 1) / 12
    return early_acc, early_exit

def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.get_parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criteria = torch.nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(10):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            train_step(model, input_data=(audio.to(device), image.to(device)), optimizer=optimizer,
                       criteria=criteria, label=text.to(device))
        scheduler.step()
        model.eval()
        acc = []
        exits = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a, e, = test_step(model, input_data=(audio.to(device), image.to(device)), label=text)
                acc.append(a)
                exits.append(e)
        acc = np.mean(acc, axis=0)
        exits = np.mean(exits, axis=0)
        print('epoch', epoch)
        print('accuracy for each threshold:', acc)
        print('computation for each threshold:', exits)
        if np.mean(acc) > best_acc:
            best_acc = np.mean(acc)
            torch.save(model.state_dict(), 'exit_' + args.task + '_' + str(epoch) + '_' + str(acc[-1]) + '.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='train')
    parser.add_argument('-w', '--worker', default=4, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    args = parser.parse_args()
    workers = args.worker
    batch_size = args.batch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    model = AVnet_Exit().to(device)
    model.audio.load_state_dict(torch.load('A_6_0.5303089942924621.pth'))
    model.image.load_state_dict(torch.load('V_7_0.5041330446762449.pth'))
    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    if args.task == 'train':
        train(model, train_dataset, test_dataset)
    else:
        profile(model, test_dataset)

