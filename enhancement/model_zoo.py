import matplotlib.pyplot as plt
import torch
import numpy as np
import soundfile as sf
from evaluation import batch_pesq, SI_SDR, lsd, batch_stoi, eval_ASR
import torch.nn.functional as F
from scipy import signal
from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM
from audio_zen.acoustics.feature import drop_band, stft, istft
from torch.cuda.amp import autocast
from pathlib import Path
from speechbrain.pretrained import EncoderDecoderASR
'''
This script contains 4 model's training and test due to their large differences (for concise)
1. FullSubNet, LSTM, spectrogram magnitude -> cIRM
2. SEANet, GAN, time-domain, encoder-decoder,
3. A2Net (VibVoice), spectrogram magnitude -> spectrogram magnitude
4. Conformer, GAN, spectrogram real+imag -> real+imag
'''
seg_len_mic = 512
overlap_mic = 256
seg_len_imu = 64
overlap_imu = 32
rate_mic = 16000
rate_imu = 1600
freq_bin_high = 8 * int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1

# Uncomment for using another pre-trained model
# asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
#                                            savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
#                                            run_opts={"device": "cuda"})
def eval(clean, predict, text=None):
    if text is not None:
        wer_clean, wer_noisy = eval_ASR(clean, predict, text, asr_model)
        metrics = [wer_clean, wer_noisy]
    else:
        metric1 = batch_pesq(clean, predict, 'wb')
        metric2 = batch_pesq(clean, predict, 'nb')
        metric3 = SI_SDR(clean, predict)
        #metric3 = lsd(clean, predict)
        metric4 = batch_stoi(clean, predict)
        metrics = [metric1, metric2, metric3, metric4]
    return np.stack(metrics, axis=1)
def Spectral_Loss(x_mag, y_mag):
    """Calculate forward propagation.
          Args:
              x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
              y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
          Returns:
              Tensor: Spectral convergence loss value.
          """
    x_mag = torch.clamp(x_mag, min=1e-7)
    y_mag = torch.clamp(y_mag, min=1e-7)
    spectral_convergenge_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
    log_stft_magnitude = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
    return 0.5 * spectral_convergenge_loss + 0.5 * log_stft_magnitude
def train_voicefilter(model, acc, noise, clean, optimizer, device='cuda'):
    noisy_mag, _, _, _ = stft(noise, 1200, 160, 400)
    clean_mag, _, _, _ = stft(clean, 1200, 160, 400)
    optimizer.zero_grad()

    noisy_mag = noisy_mag.to(device=device)
    clean_mag = clean_mag.to(device=device)
    mask = model(noisy_mag.permute(0, 2, 1), acc.to(device=device)).permute(0, 2, 1)
    predict = noisy_mag * mask

    loss = F.mse_loss(predict, clean_mag)
    loss.backward()
    optimizer.step()
    return loss.item()
def test_voicefilter(model, acc, noise, clean, device='cuda', text=None, data=False):
    noisy_mag, noisy_phase, _, _ = stft(noise, 1200, 160, 400)

    noisy_mag = noisy_mag.to(device=device)
    mask = model(noisy_mag.permute(0, 2, 1), acc.to(device=device)).permute(0, 2, 1)
    predict = noisy_mag * mask

    predict = predict.cpu()
    predict = istft((predict, noisy_phase), 1200, 160, 400, input_type="mag_phase").numpy()
    clean = clean.numpy()
    return eval(clean, predict, text=text)

def train_vibvoice(model, acc, noise, clean, optimizer, device='cuda'):
    noisy_mag, _, _, _ = stft(noise, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    optimizer.zero_grad()
    # VibVoice
    noisy_mag = noisy_mag.to(device=device)
    clean_mag = clean_mag.to(device=device)
    clean_mag = torch.unsqueeze(clean_mag[:, 1:257, 1:], 1)
    clean, acc = model(noisy_mag, acc)

    loss = Spectral_Loss(clean, clean_mag)
    loss += 0.1 * F.mse_loss(acc, clean_mag[:, :, :32, :])
    loss.backward()

    for name, param in model.named_parameters():
        if torch.isnan(param.grad).any():
            print("nan gradient found")
            raise SystemExit

    optimizer.step()
    return loss.item()
def test_vibvoice(model, acc, noise, clean, device='cuda', text=None, data=False):

    noisy_mag, noisy_phase, _, _ = stft(noise, 640, 320, 640)
    # VibVoice
    noisy_mag = noisy_mag.to(device=device)

    predict, acc = model(noisy_mag, acc)
    predict = predict.squeeze(1).cpu()
    predict = F.pad(predict, (1, 0, 1, 321 - 257))
    predict = istft((predict, noisy_phase), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy()
    if data:
        noise = noise.squeeze(1).numpy()
        noise = np.pad(noise, ((0, 0), (1, 321-257), (1, 0)))
        noise = signal.istft(noise, 16000, nperseg=640, noverlap=320)[-1]
        return eval(clean, predict, text=text), predict, noise
    else:
        return eval(clean, predict, text=text)
def train_fullsubnet(model, acc, noise, clean, optimizer, device='cuda'):
    optimizer.zero_grad()
    noise = noise.to(device=device)
    clean = clean.to(device=device)

    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noise, 640, 320, 640)
    _, _, clean_real, clean_imag = stft(clean, 640, 320, 640)
    cIRM = build_complex_ideal_ratio_mask(noisy_real, noisy_imag, clean_real, clean_imag)  # [B, F, T, 2]
    #cIRM = drop_band(cIRM, 2)

    noisy_mag = noisy_mag.unsqueeze(1)
    cRM = model(noisy_mag)
    loss = F.mse_loss(cIRM, cRM)

    loss.backward()
    optimizer.step()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 10
    )
    return loss.item()

def test_fullsubnet(model, acc, noise, clean, device='cuda', text=None, data=False):
    noisy_mag, _, noisy_real, noisy_imag = stft(noise, 512, 256, 512)
    noisy_mag = noisy_mag.to(device=device).unsqueeze(1)
    predict = model(noisy_mag)
    cRM = decompress_cIRM(predict.permute(0, 2, 3, 1)).cpu()
    enhanced_real = cRM[..., 0] * noisy_real - cRM[..., 1] * noisy_imag
    enhanced_imag = cRM[..., 1] * noisy_real + cRM[..., 0] * noisy_imag
    predict = istft((enhanced_real, enhanced_imag), 512, 256, 512, length=noise.size(-1), input_type="real_imag").numpy()
    clean = clean.numpy()
    # clean = clean / np.max(clean) * 0.8
    # predict = predict / np.max(predict) * 0.8
    if data:
        noise = noise.squeeze(1).numpy()
        noise = np.pad(noise, ((0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
        noise = signal.istft(noise, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)[-1]
        return eval(clean, predict, text=text), predict, noise
    else:
        return eval(clean, predict, text=text)
def train_SEANet(model, acc, noise, clean, optimizer, optimizer_disc=None, discriminator=None, device='cuda'):
    predict1, predict2 = model(acc.to(device=device, dtype=torch.float), noise.to(device=device, dtype=torch.float))
    # without discrinimator
    if discriminator is None:
        loss = F.mse_loss(torch.unsqueeze(predict1, 1), clean.to(device=device, dtype=torch.float))
        loss.backward()
        optimizer.step()
        return loss.item()
    else:
        # generator
        optimizer.zero_grad()
        disc_fake = discriminator(predict1)
        disc_real = discriminator(clean.to(device=device, dtype=torch.float))
        loss = 0
        for (feats_fake, score_fake), (feats_real, _) in zip(disc_fake, disc_real):
            loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
            for feat_f, feat_r in zip(feats_fake, feats_real):
                loss += 100 * torch.mean(torch.abs(feat_f - feat_r))
                #loss += 100 * F.mse_loss(feat_f, feat_r)
        loss.backward()
        optimizer.step()

        # discriminator
        optimizer_disc.zero_grad()
        disc_fake = discriminator(predict1.detach())
        disc_real = discriminator(clean.to(device=device, dtype=torch.float))
        discrim_loss = 0
        for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
            discrim_loss += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
            discrim_loss += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))
        discrim_loss.backward()
        optimizer_disc.step()
        return loss.item(), discrim_loss.item()
def test_SEANet(model, acc, noise, clean, device='cuda', text=None):
    predict1, predict2 = model(acc.to(device=device, dtype=torch.float), noise.to(device=device, dtype=torch.float))
    clean = clean.squeeze(1).numpy()
    predict = predict1.cpu().numpy()
    return eval(clean, predict, text)

def train_conformer(model, acc, noise, clean, optimizer, optimizer_disc=None, discriminator=None, device='cuda'):

    clean_mag = clean.abs().to(device=device, dtype=torch.float)
    # predict real and imag - conformer
    optimizer.zero_grad()
    noisy_spec = torch.stack([noise.real, noise.imag], 1).to(device=device, dtype=torch.float).permute(0, 1, 3, 2)
    clean_real, clean_imag = clean.real.to(device=device, dtype=torch.float), clean.imag.to(device=device, dtype=torch.float)
    est_real, est_imag = model(noisy_spec)
    est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
    loss = 0.9 * F.mse_loss(est_mag, clean_mag) + 0.1 * F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

    # adversarial training
    BATCH_SIZE = noise.shape[0]
    one_labels = torch.ones(BATCH_SIZE).cuda()
    predict_fake_metric = discriminator(clean_mag, est_mag)
    gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
    loss += 0.1 * gen_loss_GAN
    loss.backward()
    optimizer.step()

    # discriminator loss
    optimizer_disc.zero_grad()
    predict = torch.complex(est_real, est_imag)
    predict_audio = predict.detach().cpu().numpy()
    predict_audio = np.pad(predict_audio, ((0, 0), (0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
    predict_audio = signal.istft(predict_audio, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)[-1]

    clean_audio = np.pad(clean, ((0, 0), (0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
    clean_audio = signal.istft(clean_audio, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)[-1]

    pesq_score = discriminator.batch_pesq(clean_audio, predict_audio)
    # The calculation of PESQ can be None due to silent part
    if pesq_score is not None:
        optimizer_disc.zero_grad()
        predict_enhance_metric = discriminator(clean_mag, predict.detach())
        predict_max_metric = discriminator(clean_mag, clean_mag)
        discrim_loss = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                       F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        discrim_loss.backward()
        optimizer_disc.step()
    else:
        discrim_loss = torch.tensor([0.])
    return loss.item(), discrim_loss.item()
def test_conformer(model, acc, noise, clean, device='cuda'):
    # predict real and imag - conformer
    noisy_spec = torch.stack([noise.real, noise.imag], 1).to(device=device, dtype=torch.float).permute(0, 1, 3, 2)
    est_real, est_imag = model(noisy_spec)
    predict = torch.complex(est_real, est_imag)

    predict = predict.cpu().numpy()
    clean = clean.cpu().numpy()

    predict = np.pad(predict, ((0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
    predict = signal.istft(predict, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)[-1]

    clean = np.pad(clean, ((0, 0), (1, int(seg_len_mic / 2) + 1 - freq_bin_high), (1, 0)))
    clean = signal.istft(clean, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)[-1]
    return eval(clean, predict)
