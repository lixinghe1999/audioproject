import numpy as np
import torch
def training_step(model, batch, optimizer, device):
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

    # enable fbsp-parameters
    for p in model.audio.fbsp.parameters():
        p.requires_grad = True

    # enable logit scaling
    model.logit_scale_ai.requires_grad = True
    model.logit_scale_at.requires_grad = True

    # add fbsp- and logit scaling parameters to a separate group without weight decay
    param_groups.append({
        'params': [
                      p for p in model.audio.fbsp.parameters()
                  ] + [
                      model.logit_scale_ai,
                      model.logit_scale_at
                  ],
        'weight_decay': 0.0
    })
    return model, param_groups
def zero_shot_eval(y_pred, y, class_idx_to_label, print_result=False):
    # calculate model confidence
    num_audio = y_pred.shape[0]
    top1_a = 0; top3_a = 0; log = []
    for audio_idx in range(num_audio):
        # acquire Top-3 most similar results
        conf_values, ids = y_pred[audio_idx].topk(3)
        gt = y[audio_idx].item()
        if gt == ids[0]:
            top1_a += 1
        else:
            log.append([class_idx_to_label[gt], class_idx_to_label[ids[0].item()]])
        if (gt == ids).any():
            top3_a += 1
        # format output strings
        if print_result:
            query = class_idx_to_label[gt]
            results = ', '.join([f'{class_idx_to_label[i.item()]:>15s} ({v:06.2%})' for v, i in zip(conf_values, ids)])
            print(query + ', ' + results)
    return top1_a/num_audio, top3_a/num_audio, log
def eval_step(batch, model, dataset, device, text_features=None, save=None):
    model.eval()
    with torch.no_grad():
        audio, image, text = batch
        if audio is not None:
            audio = audio.to(device)
        if image is not None:
            image = image.to(device)

        if text_features is not None:
            ((audio_features, image_features, _), _), _ = model(audio=audio, image=image,
             batch_indices= torch.arange(len(text), dtype=torch.int64, device=device))
            class_idx_to_label = dataset.class_idx_to_label
            label_to_class_idx = dataset.label_to_class_idx
            y = torch.zeros(len(text), len(class_idx_to_label), dtype=torch.int8, device=device)
            for item_idx, labels in enumerate(text):
                class_ids = list(sorted([
                    label_to_class_idx[lb] for lb in labels]))
                y[item_idx][class_ids] = 1
        else:
            ((audio_features, image_features, text_features), _), _ = model(audio=audio, image=image, text=text, batch_indices=
            torch.arange(len(text), dtype=torch.int64, device=device))
            text_features = text_features.unsqueeze(1).transpose(0, 1)
            y = torch.eye(len(text))

        if audio is not None:
            audio_features = audio_features.unsqueeze(1)
            y_pred_a = (audio_features @ text_features.transpose(-1, -2)).squeeze(1).cpu()
        else:
            audio_features = torch.zeros((len(text), 1024))
            y_pred_a = None
        if image is not None:
            image_features = image_features.unsqueeze(1)
            y_pred_i = (image_features @ text_features.transpose(-1, -2)).squeeze(1).cpu()
        else:
            image_features = torch.zeros((len(text), 1024))
            y_pred_i = None

        if save is not None:
            print(text_features.shape, audio_features.shape, image_features.shape)
            save['text'].append(text_features.squeeze(0).cpu().numpy())
            save['audio'].append(audio_features.squeeze(0).cpu().numpy())
            save['image'].append(audio_features.squeeze(0).cpu().numpy())
            save['y'].append(y.cpu().numpy())
        y = y.argmax(dim=-1)
    return y_pred_a, y_pred_i, y
def validate_one_model(model, dataset, text_features, device):
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=16, shuffle=False,
                                              drop_last=False, collate_fn=collate_fn)
    acc_1 = 0
    for batch in loader:
        y_pred, y = eval_step(batch, model, text_features, dataset, device)
        top1, top3, log = zero_shot_eval(y_pred, y, dataset.class_idx_to_label, print_result=False)
        acc_1 += top1
    return acc_1 / len(loader)