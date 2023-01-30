import numpy as np
import torch
from model import AudioCLIP
from utils.datasets.esc50 import ESC50
from utils.datasets.us8k import UrbanSound8K
from utils.train import collate_fn, zero_shot_eval, eval_step


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    # MODEL_FILENAME = 'full[0.95, 0.995].pth'
    SAMPLE_RATE = 44100

    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    dataset = ESC50('../dataset/ESC50', fold=1, train=False, sample_rate=SAMPLE_RATE, length=5)
    #dataset = UrbanSound8K('../dataset/UrbanSound8K', fold=1, train=False, sample_rate=SAMPLE_RATE, length=4)
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=16, shuffle=False, drop_last=False,
                                         collate_fn=collate_fn)
    acc_1 = 0
    acc_3 = 0
    logs = []
    # We only compute the text features once
    ((_, _, text_features), _), _ = model(text=[
        [dataset.class_idx_to_label[class_idx]]
        for class_idx in sorted(dataset.class_idx_to_label.keys())
    ], batch_indices=torch.arange(len(dataset.class_idx_to_label), dtype=torch.int64, device=device))
    text_features = text_features.unsqueeze(1).transpose(0, 1)
    save = []
    for batch in loader:
        y_pred, y = eval_step(batch, model, text_features, dataset, device, save)
        print(np.concatenate(save).shape)
        top1, top3, log = zero_shot_eval(y_pred, y, dataset.class_idx_to_label, print_result=False)
        acc_1 += top1
        acc_3 += top3
        logs += log
    np.savez('save_embedding', audio=np.concatenate(save), text=text_features.cpu().numpy())
    print(acc_1/len(loader), acc_3/len(loader))
