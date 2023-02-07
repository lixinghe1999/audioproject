from utils.datasets.ucf101 import UCF101
from utils.datasets.vggsound import VGGSound
import numpy as np
import torch
from model import AudioCLIP
from utils.train import collate_fn, zero_shot_eval, eval_step
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
# remove annoying librosa warning

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    # MODEL_FILENAME = 'AudioCLIP-Partial-Training.pt'
    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    # model = AudioCLIP().to(device)
    # dataset = UCF101()
    dataset = VGGSound()
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=16, shuffle=True,
                                         drop_last=True, pin_memory=True)
    acc_a = []
    acc_i = []
    save = {'audio': [], 'image': [], 'text': [], 'y': [], 'name': []}
    with torch.no_grad():
        ((_, _, text_features), _), _ = model(text=[
            [dataset.class_idx_to_label[class_idx]]
            for class_idx in sorted(dataset.class_idx_to_label.keys())
        ], batch_indices=torch.arange(len(dataset.class_idx_to_label), dtype=torch.int64, device=device))
        text_features = text_features.unsqueeze(1).transpose(0, 1)
    print('finish text embedding extraction')
    for batch in tqdm(loader):
        y_pred_a, y_pred_i, y = eval_step(batch, model, dataset, device, save=save, text_features=text_features)
        top1, top3 = zero_shot_eval(y_pred_a, y)
        print(top1, top3)
        acc_a.append([top1, top3])
        top1, top3 = zero_shot_eval(y_pred_i, y)
        print(top1, top3)
        acc_i.append([top1, top3])
    print(np.mean(acc_a, axis=0))
    print(np.mean(acc_i, axis=0))
    np.savez('save_embedding', audio=np.concatenate(save['audio']), image=np.concatenate(save['image']),
            text=text_features.squeeze(0).cpu().numpy(), y=np.concatenate(save['y']), name=np.concatenate(save['name']))
