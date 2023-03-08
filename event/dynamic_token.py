import time
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model.dyvit import AVnet_Dynamic, AudioTransformerDiffPruning, VisionTransformerDiffPruning
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
    output = model(*input_data)
    if isinstance(output, tuple):
        output = output[0]
    acc = (torch.argmax(output, dim=-1).cpu() == label).sum()/len(label)
    return acc.item()
def profile(model, test_dataset):
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=32, shuffle=False)
    model.eval()
    token_ratio = [[0.8, 0.8**2, 0.8**3], [0.7, 0.7**2, 0.7**3], [0.6, 0.6**2, 0.6**3], [0.75, 0.5, 0.25]]
    acc = []
    with torch.no_grad():
        for ratio in token_ratio:
            model.token_ratio = ratio
            for batch in tqdm(test_loader):
                audio, image, text, _ = batch
                a = test_step(model, input_data=[audio.to(device), image.to(device)], label=text)
                acc.append(a)
        mean_acc = np.mean(acc)
        print('preserved ratio', ratio)
        print('accuracy:', mean_acc)
def train(model, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True,
                                               drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=workers, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.00001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    best_acc = 0
    for epoch in range(10):
        # model.train()
        # for idx, batch in enumerate(tqdm(train_loader)):
        #     audio, image, text, _ = batch
        #     train_step(model, input_data=[audio.to(device), image.to(device)], optimizer=optimizer,
        #                    criteria=criteria, label=text.to(device))
        # scheduler.step()
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)

    pruning_loc = (3, 6, 9)
    base_rate = 0.7
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
    # config_small = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    #                     pruning_loc=pruning_loc, token_ratio=token_ratio)
    # config_base = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    #                    pruning_loc=pruning_loc, token_ratio=token_ratio)


    model = AVnet_Dynamic(pruning_loc=pruning_loc, token_ratio=token_ratio, pretrained=False, distill=True).to(device)
    # model.audio.load_state_dict(torch.load('token_network/A_6_0.5303089942924621.pth'), strict=False)
    # model.image.load_state_dict(torch.load('token_network/V_7_0.5041330446762449.pth'), strict=False)
    model.load_state_dict(torch.load('token_network/AV_6_0.6778193269041527.pth'), strict=False)

    # model = VisionTransformerDiffPruning(pruning_loc=pruning_loc, token_ratio=token_ratio).to(device)
    # model.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'], strict=False)

    # model = AudioTransformerDiffPruning(config_small, imagenet_pretrain=False).to(device)

    dataset = VGGSound()
    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len_test], generator=torch.Generator().manual_seed(42))

    if args.task == 'train':
        criteria = torch.nn.CrossEntropyLoss()
        train(model, train_dataset, test_dataset)
    elif args.task == 'distill':
        teacher_model = AVnet_Dynamic(pruning_loc=(), pretrained=False, distill=True).to(device)
        teacher_model.load_state_dict(torch.load('token_network/AV_6_0.6778193269041527.pth'), strict=False)
        teacher_model.eval()
        criteria = DistillDiffPruningLoss_dynamic(teacher_model, torch.nn.CrossEntropyLoss(), clf_weight=1.0,
                                                  keep_ratio=[0.75, 0.5, 0.25], mse_token=True, ratio_weight=2.0, distill_weight=0.5)
        train(model, train_dataset, test_dataset)
    elif args.task == 'profile':
        profile(model, test_dataset)

