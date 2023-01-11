import torch
from model import AudioCLIP
from utils.datasets.esc50 import ESC50
from utils.datasets.split_dataset import split_dataset, split_dataset_type, split_type, split_type_random, find_base_class_random
from utils.train import training_step, prepare_model, collate_fn, zero_shot_eval, eval_step
import numpy as np
def fine_tune(tr, te, MODEL_FILENAME, test_dataset, text_features, device):
    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    model, param_groups = prepare_model(model)
    train_loader = torch.utils.data.DataLoader(dataset=tr, num_workers=4, batch_size=16, shuffle=True,
                                               drop_last=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=te, num_workers=4, batch_size=16, shuffle=False,
                                              drop_last=False, collate_fn=collate_fn)
    optimizer = torch.optim.SGD(param_groups, **{**{
        "lr": 5e-5, "momentum": 0.9, "nesterov": True, "weight_decay": 5e-4}, **{'lr': 5e-5}})
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    for e in range(10):
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
        if e == 0:
            loss_best = mean_lost
        if mean_lost <= loss_best:
            ckpt_best = model.audio.state_dict()
            loss_best = mean_lost
            metric_best = metric
    print('the best result for one user:', metric_best)
    # torch.save(ckpt_best, 'assets/' + str(metric_best) + '.pt')
    return metric_best
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    SAMPLE_RATE = 44100
    train_dataset = ESC50('../dataset/ESC50', fold=1, train=True, sample_rate=SAMPLE_RATE, few_shot=3)
    test_dataset = ESC50('../dataset/ESC50', fold=1, train=False, sample_rate=SAMPLE_RATE)

    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    model.eval()
    ((_, _, text_features), _), _ = model(text=[
        [test_dataset.class_idx_to_label[class_idx]]
        for class_idx in sorted(test_dataset.class_idx_to_label.keys())
    ], batch_indices=torch.arange(len(test_dataset.class_idx_to_label), dtype=torch.int64, device=device))
    text_features = text_features.unsqueeze(1).transpose(0, 1)
    # num_users = 8
    # train_dataset_list, type_list = split_dataset(train_dataset, num_users)
    # test_dataset_list = split_dataset_type(test_dataset, type_list)
    num_users = 10
    num_base = 3
    num_repeat = 10
    metric = []
    type_list = split_type_random(train_dataset.class_idx_to_label, num_users, 3)
    type_list = find_base_class_random(train_dataset.class_idx_to_label, type_list, num_repeat, num_base)
    train_dataset_list = split_dataset_type(train_dataset, type_list)
    test_dataset_list = split_dataset_type(test_dataset, type_list)
    for tr, te in zip(train_dataset_list, test_dataset_list):
        metric_best = fine_tune(tr, te, MODEL_FILENAME, test_dataset, text_features, device)
        metric.append(metric_best)
    metric = np.stack(metric)
    np.savez('user_specific', type_list=type_list, metric=metric, allow_pickle=True)
    print('mean top1, top3 accuracy', np.mean(metric, axis=0))

