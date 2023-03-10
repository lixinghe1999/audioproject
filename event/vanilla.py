import torchvision.models
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.vanilla_model import AVnet
from model.vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning
import argparse
torchvision.models.resnet50()
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
# remove annoying librosa warning
def step(model, input_data, optimizer, criteria, label):
    audio, image = input_data
    # Track history only in training
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
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=4, shuffle=False)
    best_acc = 0
    if args.task == 'AV':
        # for param in model.audio.parameters():
        #     param.requires_grad = False
        # for param in model.image.parameters():
        #     param.requires_grad = False
        optimizer = torch.optim.Adam(model.fusion_parameter(), lr=.0001, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criteria = torch.nn.CrossEntropyLoss()
    for epoch in range(20):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            audio, image, text, _ = batch
            loss = step(model, input_data=(audio.to(device), image.to(device)), optimizer=optimizer,
                        criteria=criteria, label=text.to(device))
        scheduler.step()
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
            torch.save(model.state_dict(), 'vanilla_' + args.task + '_' + str(epoch) + '_' + str(np.mean(acc)) + '.pth')
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
    config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                  pruning_loc=())
    embed_dim = 768
    if args.task == 'AV':
        model = AVnet().to(device)
        model.audio.load_state_dict(torch.load('vanilla_A_6_0.5303089942924621.pth'))
        model.image.load_state_dict(torch.load('vanilla_V_7_0.5041330446762449.pth'))
    elif args.task == 'A':
        model = AudioTransformerDiffPruning(config, imagenet_pretrain=True).to(device)
    else:
        model = VisionTransformerDiffPruning(**config).to(device)
        model.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'], strict=False)

    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))
    train(model, train_dataset, test_dataset)

