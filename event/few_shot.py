import torch
from model import AudioCLIP
from utils.datasets.esc50 import ESC50
from zero_shot import zero_shot_eval, eval_step
import numpy as np
from tqdm import tqdm
def training_step(model, batch, optimizer):
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
    batch_audio, batch_text = zip(*batch)

    if not all(audio is None for audio in batch_audio):
        batch_audio = torch.stack(batch_audio)
    else:
        batch_audio = None
    if not all(text is None for text in batch_text):
        batch_text = [idx for idx in batch_text]
    else:
        batch_text = None
    return batch_audio, batch_text

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    MODEL_FILENAME = 'ESC50_Multimodal-Audio_ACLIP-CV1_ACLIP-CV1_performance=0.9550.pt'
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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    # disable all parameters
    for p in model.parameters():
        p.requires_grad = False
    # enable only audio-related parameters
    for p in model.module.audio.parameters():
        p.requires_grad = True


    loss_best = 1
    for e in range(20):
        Loss_list = []
        for i, batch in tqdm(enumerate(train_loader)):
            loss = training_step(model, batch, optimizer)
            if i % 200 == 0 and i != 0:
                print(loss)
            Loss_list.append(loss)
        mean_lost = np.mean(Loss_list)
        scheduler.step()
        acc_1 = 0; acc_3 = 0
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
