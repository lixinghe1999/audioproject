from utils.datasets.epic_kitchen import EPIC_Kitchen
import numpy as np
import torch
from model import AudioCLIP
from utils.train import collate_fn, zero_shot_eval, eval_step


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    SAMPLE_RATE = 44100

    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    dataset = EPIC_Kitchen()
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=16, shuffle=False,
                                         drop_last=False, collate_fn=collate_fn)
    acc_a = []
    acc_i = []
    logs = []
    # We only compute the text features once
    # with torch.no_grad():
    #     ((_, _, text_features), _), _ = model(text=[
    #         [dataset.class_idx_to_label[class_idx]]
    #         for class_idx in sorted(dataset.class_idx_to_label.keys())
    #     ], batch_indices=torch.arange(len(dataset.class_idx_to_label), dtype=torch.int64, device=device))
    #     text_features = text_features.unsqueeze(1).transpose(0, 1)
    save = None
    for batch in loader:
        y_pred_a, y_pred_i, y = eval_step(batch, model, dataset, device, save=save)
        top1, top3, log = zero_shot_eval(y_pred_a, y, dataset.class_idx_to_label, print_result=False)
        print(top1, top3)
        acc_a.append([top1, top3])
        top1, top3, log = zero_shot_eval(y_pred_i, y, dataset.class_idx_to_label, print_result=False)
        print(top1, top3)
        acc_i.append([top1, top3])
        logs += log
    print(np.mean(acc_a, axis=0))
    print(np.mean(acc_i, axis=0))
    # save = np.concatenate(save)
    # np.savez('save_embedding', audio=save[:, :1024], text=text_features.cpu().numpy(), y=save[:, 1024:])
