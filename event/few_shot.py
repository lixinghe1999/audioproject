import torch
from model import AudioCLIP
from utils.datasets.esc50 import ESC50
from utils.datasets.fsd50k import FSD50K
from utils.train import training_step, prepare_model, collate_fn, zero_shot_eval, eval_step
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    # derived from ESResNeXt
    SAMPLE_RATE = 44100
    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    # train_dataset = ESC50('../dataset/ESC50', fold=1, train=True, sample_rate=SAMPLE_RATE, few_shot=1)
    # test_dataset = ESC50('../dataset/ESC50', fold=1, train=False, sample_rate=SAMPLE_RATE)
    train_dataset = FSD50K('../dataset/FSD50K', train=True, sample_rate=SAMPLE_RATE)
    test_dataset = FSD50K('../dataset/FSD50K', train=False, sample_rate=SAMPLE_RATE)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=16, shuffle=True,
                                               drop_last=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=16, shuffle=False,
                                         drop_last=False, collate_fn=collate_fn)
    model, param_groups = prepare_model(model)
    optimizer = torch.optim.SGD(param_groups, **{**{
     "lr": 5e-5, "momentum": 0.9, "nesterov": True, "weight_decay": 5e-4}, **{'lr': 5e-5 }})
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    loss_best = 2
    ((_, _, text_features), _), _ = model(text=[
        [test_dataset.class_idx_to_label[class_idx]]
        for class_idx in sorted(test_dataset.class_idx_to_label.keys())
    ], batch_indices=torch.arange(len(test_dataset.class_idx_to_label), dtype=torch.int64, device=device))
    text_features = text_features.unsqueeze(1).transpose(0, 1)
    for e in range(20):
        Loss_list = []
        for i, batch in enumerate(train_loader):
            loss = training_step(model, batch, optimizer, device)
            Loss_list.append(loss)
        mean_lost = np.mean(Loss_list)
        scheduler.step()
        acc_1 = 0; acc_3 = 0
        for batch in test_loader:
            y_pred, y = eval_step(batch, model, text_features, test_dataset, device)
            top1, top3, log = zero_shot_eval(y_pred, y, test_dataset.class_idx_to_label, print_result=False)
            acc_1 += top1
            acc_3 += top3
        metric = [acc_1 / len(test_loader), acc_3 / len(test_loader)]
        print('epoch ' + str(e) + ' ', metric, mean_lost)
        if mean_lost < loss_best:
            ckpt_best = model.audio.state_dict()
            loss_best = mean_lost
            metric_best = metric
    torch.save(ckpt_best, 'assets/' + str(metric_best) + '.pt')
