import torchvision.models

from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.vanilla_model import AVnet, SingleNet
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
        output = model(audio)
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
def train(train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=16, batch_size=64, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=16, batch_size=64, shuffle=False)
    # optimizers = [torch.optim.Adam(model.get_image_params(), lr=.0001, weight_decay=1e-4),
    #               torch.optim.Adam(model.get_audio_params(), lr=.0001, weight_decay=1e-4)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)]
    criteria = torch.nn.CrossEntropyLoss()
    for e in range(20):
        model.train()
        if e % 3 == 0 and e > 0:
            update_lr(optimizers[0], multiplier=.2)
            # update_lr(optimizers[1], multiplier=.1)
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            loss = step(model, input_data=(audio.to(device), image.to(device)), optimizers=optimizers, criteria=criteria, label=text.to(device))
        model.eval()
        acc = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                predict = model(audio.to(device))
                acc.append((torch.argmax(predict, dim=-1).cpu() == text).sum() / len(text))
        print('epoch', e, np.mean(acc))
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    # model = AVnet().to(device)
    model = SingleNet(modality='A').to(device)
    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    train(train_dataset, test_dataset)

