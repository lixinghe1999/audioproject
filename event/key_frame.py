from utils.datasets.epic_kitchen import EPIC_Kitchen
from utils.datasets.ucf101 import UCF101
import numpy as np
import torch
from model import AudioCLIP
from utils.train import collate_fn, zero_shot_eval, eval_step


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    # MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    MODEL_FILENAME = 'AudioCLIP-Partial-Training.pt'
    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    # dataset = EPIC_Kitchen()
    dataset = UCF101()
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=4, shuffle=False,
                                         drop_last=True, collate_fn=collate_fn)
    acc_a = []
    acc_i = []
    save = {'audio':[], 'image':[], 'text':[], 'y':[]}
    for batch in loader:
        y_pred_a, y_pred_i, y = eval_step(batch, model, dataset, device, save=save)
        top1, top3 = zero_shot_eval(y_pred_a, y, dataset.class_idx_to_label, print_result=False)
        print(top1, top3)
        acc_a.append([top1, top3])
        top1, top3 = zero_shot_eval(y_pred_i + y_pred_a, y, dataset.class_idx_to_label, print_result=False)
        print(top1, top3)
        acc_i.append([top1, top3])
    print(np.mean(acc_a, axis=0))
    print(np.mean(acc_i, axis=0))
    save = np.concatenate(save)
    np.savez('save_embedding', audio=save[:, :1024], image=save[:, 1024:2048], text=save[:, 2048:3072], y=save[:, 3072:])
