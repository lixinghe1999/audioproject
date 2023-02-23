import torchvision.models
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.vanilla_model import AVnet
from model.ast_vit import ASTModel, VITModel
import argparse
torchvision.models.resnet50()
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
# remove annoying librosa warning
def step(model, input_data, optimizers, criteria, label):
    audio, image = input_data
    # Track history only in training
    for branch in [0]:
        optimizer = optimizers[branch]
        if args.task == 'AV':
            output = model(audio, image)
        elif args.task == 'A':
            output = model(audio)
        else:
            output = model(image)
        # Backward
        optimizer.zero_grad()
        loss = criteria(output, label)
        loss.backward()
        optimizer.step()
    return loss
def update_lr(optimizer, multiplier = .1):
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = param_group['lr'] * multiplier
    optimizer.load_state_dict(state_dict)
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=64, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=4, shuffle=False)
    best_acc = 0
    for param in model.audio.parameters():
        param.requires_grad = False
    for param in model.image.parameters():
        param.requires_grad = False
    optimizers = [torch.optim.Adam(model.fusion_parameter(), lr=.0001, weight_decay=1e-4)]
    criteria = torch.nn.CrossEntropyLoss()
    for epoch in range(20):
        model.train()
        if epoch % 4 == 0 and epoch > 0:
            update_lr(optimizers[0], multiplier=.2)
        # for idx, batch in enumerate(tqdm(train_loader)):
        #     audio, image, text, _ = batch
        #     loss = step(model, input_data=(audio.to(device), image.to(device)), optimizers=optimizers, criteria=criteria, label=text.to(device))
        model.eval()
        acc = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                if args.task == 'AV':
                    predict = model(audio.to(device), image.to(device))
                elif args.task == 'A':
                    predict = model(audio.to(device))
                else:
                    predict = model(image.to(device))
                acc.append((torch.argmax(predict, dim=-1).cpu() == text).sum() / len(text))
        print('epoch', epoch, np.mean(acc))
        if np.mean(acc) > best_acc:
            best_acc = np.mean(acc)
            torch.save(model.state_dict(), args.task + '_' + str(epoch) + '_' + str(np.mean(acc)) + '.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    if args.task == 'AV':
        model = AVnet().to(device)
        model.load_state_dict(torch.load('AV_9_0.64882076.pth'))
        # model.audio.load_state_dict(torch.load('A_4_0.5673682.pth'))
        # model.image.load_state_dict(torch.load('V_6_0.5151336.pth'))
    elif args.task == 'A':
        model = ASTModel(input_tdim=384, audioset_pretrain=False, verbose=True, model_size='base224').to(device)
    else:
        model = VITModel(model_size='base224').to(device)
    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    train(model, train_dataset, test_dataset)

