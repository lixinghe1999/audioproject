#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:12:22 2020

@author: darp_lord
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from computing.dataset import AudioDataset
from model.identification import VGGM

transformers = transforms.ToTensor()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul(100.0 / batch_size)).item())
        return res

def test(model, Dataloader):
    counter=0
    top1=0
    top5=0
    for audio, labels in Dataloader:
        audio = audio.to(device)
        labels = labels.to(device)
        outputs = model(audio)
        corr1, corr5=accuracy(outputs, labels, topk=(1,5))
        # Cumulative values
        top1+=corr1
        top5+=corr5
        counter+=1
    print("Cumulative Val:\nTop-1 accuracy: %.5f\nTop-5 accuracy: %.5f"%(top1/counter, top5/counter))
    return top1/counter, top5/counter


if __name__=="__main__":
    LR = 0.001
    B_SIZE = 16
    N_EPOCHS = 40
    batch_sizes={
            "train":B_SIZE,
            "val":1,
            "test":1}

    Dataloaders = {}
    dataset = AudioDataset('id.json')
    num = len(dataset)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8 * num), num - int(0.8 * num)])
    Dataloaders['train'] = DataLoader(train_set, batch_size=batch_sizes['train'], shuffle=True, num_workers=2)
    Dataloaders['val'] = DataLoader(val_set, batch_size=batch_sizes['train'], shuffle=False, num_workers=2)
    # Dataloaders['test']=[DataLoader(Datasets['test'], batch_size=batch_sizes['test'], shuffle=False)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGM(16).to(device)

    pretrain = torch.load('VGGM300_BEST_140_81.99.pth')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pretrain.items() if k.split('.')[0] == 'features'}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.99, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1/1.17)
    #Save models after accuracy crosses 75
    best_acc = 75
    update_grad = 1
    best_epoch = 0
    print("Start Training")
    for epoch in range(N_EPOCHS):
        running_loss=0.0
        corr1 = 0
        corr5 = 0
        top1 = 0
        top5 = 0
        random_subset = None
        loop = tqdm(Dataloaders['train'])
        loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
        for counter, (audio, labels) in enumerate(loop, start=1):
            optimizer.zero_grad()
            audio = audio.to(device)
            labels = labels.to(device)
            if counter==32:
                random_subset=audio
            outputs = model(audio)
            loss = loss_func(outputs, labels)
            running_loss += loss
            corr1, corr5 = accuracy(outputs, labels, topk=(1,5))
            top1+=corr1
            top5+=corr5
            if(counter%update_grad==0):
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=(running_loss.item()/(counter)), top1_acc=top1/(counter), top5_acc=top5/counter)

        with torch.no_grad():
            acc1, _ = test(model, Dataloaders['val'])
            if acc1 > best_acc:
                best_acc = acc1
                best_model = model.state_dict()
                best_epoch = epoch
                torch.save(best_model, "VGGMVAL_BEST_%d_%.2f.pth"%(best_epoch, best_acc))
        scheduler.step()
    print('Finished Training..')
