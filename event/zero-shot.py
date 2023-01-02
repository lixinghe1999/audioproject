import torch
from model import AudioCLIP
from utils.transforms import ToTensor1D
from utils.datasets.esc50 import ESC50
def zero_shot_eval(logits_audio_text, y, class_idx_to_label):
    # calculate model confidence
    num_audio = logits_audio_text.shape[0]
    confidence = logits_audio_text.softmax(dim=0)
    for audio_idx in range(num_audio):
        # acquire Top-3 most similar results
        conf_values, ids = confidence[audio_idx].topk(3)
        # format output strings
        query = class_idx_to_label[y[audio_idx].item()]
        results = ', '.join([f'{class_idx_to_label[i.item()]:>15s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

        print(query + ', ' + results)
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
    torch.cuda.set_device(1)
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    # derived from ESResNeXt
    SAMPLE_RATE = 44100

    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    audio_transforms = ToTensor1D()
    dataset = ESC50('../dataset/ESC50', train=False, transform_audio=audio_transforms, sample_rate=SAMPLE_RATE)
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=6, shuffle=True, collate_fn=collate_fn)

    with torch.no_grad():
        for sample in loader:
            audio, text = sample
            audio = audio.to(device)
            print(audio.shape, text)

            ((audio_features, _, _), _), _ = model(audio=audio, batch_indices=
            torch.arange(audio.shape[0], dtype=torch.int64, device=device))

            ((_, _, text_features), _), _ = model(text=[
                        [dataset.class_idx_to_label[class_idx]]
                        for class_idx in sorted(dataset.class_idx_to_label.keys())
                    ], batch_indices=torch.arange(
                        len(dataset.class_idx_to_label), dtype=torch.int64, device=device
                    ))
            print(audio_features.shape, text_features.shape)
            audio_features = audio_features.unsqueeze(1)
            text_features = text_features.unsqueeze(1).transpose(0, 1)
            print(audio_features.shape, text_features.shape)
            logit_scale_at = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)
            print(logit_scale_at)
            y_pred = (logit_scale_at * audio_features @ text_features.transpose(-1, -2)).squeeze(1)
            y = torch.zeros(
                audio.shape[0], len(dataset.class_idx_to_label), dtype=torch.int8, device=device
            )
            for item_idx, labels in enumerate(text):
                class_ids = list(sorted([
                    dataset.label_to_class_idx[lb] for lb in labels]))
                y[item_idx][class_ids] = 1
            print(y)
            y_pred = torch.softmax(y_pred, dim=-1).cpu()
            y = y.argmax(dim=-1)
            print(y_pred.shape)
            print(y.shape)
            zero_shot_eval(y_pred, y, dataset.class_idx_to_label)
            break
