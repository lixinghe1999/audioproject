import torch
from model import AudioCLIP
from utils.transforms import ToTensor1D
from utils.datasets.esc50 import ESC50
def training_step(engine: ieng.Engine, batch) -> torch.Tensor:
    model.train()
    model.epoch = engine.state.epoch
    model.batch_idx = (engine.state.iteration - 1) % len(train_loader)
    model.num_batches = len(train_loader)

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

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    MODEL_FILENAME = 'ESC50_Multimodal-Audio_ACLIP-CV1_ACLIP-CV1_performance=0.9550.pt'
    # derived from ESResNeXt
    SAMPLE_RATE = 44100

    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
    audio_transforms = ToTensor1D()
    dataset = ESC50('../dataset/ESC50', train=False, transform_audio=audio_transforms, sample_rate=SAMPLE_RATE)
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=4, shuffle=True, drop_last=False,
                                         collate_fn=collate_fn)