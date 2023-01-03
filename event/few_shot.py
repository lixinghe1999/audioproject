import torch
from model import AudioCLIP
from utils.datasets.esc50 import ESC50
from zero_shot import zero_shot_eval, eval_step
import numpy as np
from tqdm import tqdm
def training_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad()
    audio, image, text = batch
    if audio is not None:
        audio = audio.to(device)
    if image is not None:
        image = image.to(device)

    batch_indices = torch.arange(audio.shape[0], dtype=torch.int64, device=device)
    _, loss = model(audio, image, text, batch_indices)

    if loss.ndim > 0:
        loss = loss.mean()

    loss.backward(retain_graph=False)
    optimizer.step(None)
    return loss.item()
def collate_fn(batch):
    batch_audio, batch_image, batch_text = zip(*batch)

    keep_ids = [idx for idx, (_, _) in enumerate(zip(batch_audio, batch_image))]

    if not all(audio is None for audio in batch_audio):
        batch_audio = [batch_audio[idx] for idx in keep_ids]
        batch_audio = torch.stack(batch_audio)
    else:
        batch_audio = None

    if not all(image is None for image in batch_image):
        batch_image = [batch_image[idx] for idx in keep_ids]
        batch_image = torch.stack(batch_image)
    else:
        batch_image = None

    if not all(text is None for text in batch_text):
        batch_text = [batch_text[idx] for idx in keep_ids]
    else:
        batch_text = None
    return batch_audio, batch_image, batch_text
def prepare_model(model):
    # disable all parameters
    for p in model.parameters():
        p.requires_grad = False
    # enable only audio-related parameters
    for p in model.audio.parameters():
        p.requires_grad = True
        # disable fbsp-parameters
    for p in model.audio.fbsp.parameters():
        p.requires_grad = False

    # disable logit scaling
    model.logit_scale_ai.requires_grad = False
    model.logit_scale_at.requires_grad = False

    # add only enabled parameters to optimizer's list
    param_groups = [
        {'params': [p for p in model.parameters() if p.requires_grad]}
    ]

    # # enable fbsp-parameters
    # for p in model.audio.fbsp.parameters():
    #     p.requires_grad = True
    #
    # # enable logit scaling
    # model.logit_scale_ai.requires_grad = True
    # model.logit_scale_at.requires_grad = True
    #
    # # add fbsp- and logit scaling parameters to a separate group without weight decay
    # param_groups.append({
    #     'params': [
    #                   p for p in model.audio.fbsp.parameters()
    #               ] + [
    #                   model.logit_scale_ai,
    #                   model.logit_scale_at
    #               ],
    #     'weight_decay': 0.0
    # })
    return model, param_groups
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    # derived from ESResNeXt
    SAMPLE_RATE = 44100

    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    train_dataset = ESC50('../dataset/ESC50', train=True, sample_rate=SAMPLE_RATE)
    test_dataset = ESC50('../dataset/ESC50', train=False, sample_rate=SAMPLE_RATE)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=16, shuffle=True, drop_last=False,
                                         collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=16, shuffle=False,
                                         drop_last=False, collate_fn=collate_fn)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=5e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    # model, param_groups = prepare_model(model)
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name, p.shape)
    # optimizer = torch.optim.SGD(param_groups, **{**{
    #   "lr": 5e-5, "momentum": 0.9,"nesterov": True, "weight_decay": 5e-4}, **{'lr': 5e-5 }})
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)


    loss_best = 1
    for e in range(20):
        Loss_list = []
        for i, batch in enumerate(train_loader):
            loss = training_step(model, batch, optimizer)
            if i % 200 == 0 and i != 0:
                print(loss)
            Loss_list.append(loss)
        mean_lost = np.mean(Loss_list)
        scheduler.step()
        acc_1 = 0; acc_3 = 0
        with torch.no_grad():
            for batch in test_loader:
                y_pred, y = eval_step(batch, model)
                top1, top3, log = zero_shot_eval(y_pred, y, test_dataset.class_idx_to_label, print_result=False)
                acc_1 += top1
                acc_3 += top3
            metric = [acc_1 / len(test_loader), acc_3 / len(test_loader)]
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            metric_best = metric
            torch.save(ckpt_best, 'pretrain/' + str(metric_best) + '.pth')
