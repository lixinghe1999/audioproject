
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning
from model.dynamicvit_model import AVnet_Dynamic
from utils.losses import DistillDiffPruningLoss_dynamic
import warnings
from tqdm import tqdm
import argparse
warnings.filterwarnings("ignore")

def train_step(model, input_data, optimizer, criteria, label):
    # cumulative loss
    outputs = model(*input_data)
    optimizer.zero_grad()
    if args.task == 'distill':
        loss, loss_part = criteria(input_data, outputs, label)
    else:
        loss = criteria(outputs[0], label)
    loss.backward()
    optimizer.step()
    return loss.item()
def test_step(model, input_data, label):
    outputs = model(*input_data)
    if args.task == 'profile':
        output, t, r = outputs
        acc = (torch.argmax(output, dim=-1).cpu() == label).sum() / len(label)
        return acc.item(), r
    else:
        output, feature = outputs
        acc = (torch.argmax(output, dim=-1).cpu() == label).sum()/len(label)
        return acc.item()
def profile(model, test_dataset):
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=1, shuffle=False)
    model.eval()
    token_ratio = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    acc = []
    modality_ratio = []
    with torch.no_grad():
        for ratio in token_ratio:
            model.token_ratio = [ratio, ratio**2, ratio**3]
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a, r = test_step(model, input_data=[audio.to(device), image.to(device)], label=text)
                acc.append(a)
                modality_ratio.append([r, 1-r, abs(2 * r - 1)])
            mean_acc = np.mean(acc)
            mean_ratio = np.mean(modality_ratio, axis=0)
            print('preserved ratio:', ratio)
            print('modality-wise ratio:', mean_ratio)
            print('accuracy:', mean_acc)
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    best_acc = 0
    for epoch in range(10):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            train_step(model, input_data=[audio.to(device), image.to(device)], optimizer=optimizer,
                           criteria=criteria, label=text.to(device))
        scheduler.step()
        model.eval()
        acc = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a = test_step(model, input_data=[audio.to(device), image.to(device)], label=text)
                acc.append(a)
        mean_acc = np.mean(acc)
        print('epoch', epoch)
        print('accuracy:', mean_acc)
        if mean_acc > best_acc:
            best_acc = mean_acc
            torch.save(model.state_dict(), 'dynamic_' + str(args.task) + '_' + str(epoch) + '_' + str(mean_acc) + '.pth')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)

    pruning_loc = (3, 6, 9)
    base_rate = 0.7
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]

    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))

    if args.task == 'train':
        model = AVnet_Dynamic(pruning_loc=pruning_loc, token_ratio=token_ratio, pretrained=False, distill=True).to(device)
        model.audio.load_state_dict(torch.load('vanilla_A_6_0.5303089942924621.pth'), strict=False)
        model.image.load_state_dict(torch.load('vanilla_V_7_0.5041330446762449.pth'), strict=False)

        criteria = torch.nn.CrossEntropyLoss()
        train(model, train_dataset, test_dataset)
    elif args.task == 'distill':
        model = AVnet_Dynamic(pruning_loc=pruning_loc, token_ratio=token_ratio, pretrained=False, distill=True).to(device)
        model.audio.load_state_dict(torch.load('vanilla_A_6_0.5303089942924621.pth'), strict=False)
        model.image.load_state_dict(torch.load('vanilla_V_7_0.5041330446762449.pth'), strict=False)

        teacher_model = AVnet_Dynamic(pruning_loc=(), pretrained=False, distill=True).to(device)
        teacher_model.load_state_dict(torch.load('vanilla_AV_9_0.6942149.pth'), strict=False)
        teacher_model.eval()
        criteria = DistillDiffPruningLoss_dynamic(teacher_model, torch.nn.CrossEntropyLoss(), clf_weight=1.0,
                keep_ratio=token_ratio, mse_token=True, ratio_weight=2, distill_weight=0.5)
        train(model, train_dataset, test_dataset)
    elif args.task == 'profile':
        model = AVnet_Dynamic(pruning_loc=pruning_loc, token_ratio=token_ratio, pretrained=False).to(device)
        model.load_state_dict(torch.load('dynamic_distill_9_0.6833300531391459.pth'), strict=False)
        profile(model, test_dataset)

