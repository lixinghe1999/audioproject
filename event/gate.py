import time
import pandas as pd
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.gate_model import AVnet_Gate
import warnings
from tqdm import tqdm
from datetime import date
warnings.filterwarnings("ignore")
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
def train_step(model, input_data, optimizers, criteria, label):
    audio, image = input_data
    # cumulative loss
    optimizer = optimizers[0]
    output = model(audio, image)
    optimizer.zero_grad()
    loss = criteria(output, label)
    loss.backward()
    optimizer.step()
    return loss.item()
def test_step(model, input_data, label):
    audio, image = input_data
    early_exits = np.zeros((4))
    t_start = time.time()
    output = model(audio, image)
    l = time.time() - t_start
    acc = (torch.argmax(output, dim=-1).cpu() == label).sum()/len(label)
    return acc, l
def update_lr(optimizer, multiplier = .1):
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = param_group['lr'] * multiplier
    optimizer.load_state_dict(state_dict)
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=16, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=8, batch_size=32, shuffle=False)
    optimizers = [torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)]
    criteria = torch.nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(10):
        model.train()
        model.exit = False
        if epoch % 5 == 0 and epoch > 0:
            for optimizer in optimizers:
                update_lr(optimizer, multiplier=.2)
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            train_step(model, input_data=(audio.to(device), image.to(device)), optimizers=optimizers, criteria=criteria, label=text.to(device))
        model.eval()
        acc = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a, _ = test_step(model, input_data=(audio.to(device), image.to(device)), label=text)
                acc.append(a)
        acc = np.stack(acc)
        acc = np.mean(acc, axis=0, where=acc >= 0)
        print('epoch', epoch)
        print('accuracy for early-exits:', acc)
        # best_acc = acc[-1].item()
        torch.save(model.state_dict(), str(epoch) + '_' + str(acc.item()) + '.pth')
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    model = AVnet_Gate().to(device)
    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    train(model, train_dataset, test_dataset)
    # profile(model, test_dataset)

