import librosa
import torch
import numpy as np
from evaluation import batch_pesq, SI_SDR, batch_stoi, eval_ASR, LSD
import torch.nn.functional as F
from scipy import signal
from mask import build_complex_ideal_ratio_mask, decompress_cIRM
from feature import stft, istft
from sisdr_loss import StabilizedPermInvSISDRMetric

from speechbrain.pretrained import EncoderDecoderASR
'''
This script contains 4 model's training and test due to their large differences (for concise)
1. FullSubNet, LSTM, spectrogram magnitude -> cIRM
2. SEANet, GAN, time-domain, encoder-decoder,
3. A2Net (VibVoice), spectrogram magnitude -> spectrogram magnitude
4. Conformer, GAN, spectrogram real+imag -> real+imag
'''
# Uncomment for using another pre-trained model
# asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
#                                            savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
#                                            run_opts={"device": "cuda"})
sisdr_loss = StabilizedPermInvSISDRMetric(n_actual_sources=2, n_estimated_sources=2,
                                  zero_mean=True, backward_loss=True, improvement=True)
def eval(clean, predict, text=None):
    if text is not None:
        wer_clean, wer_noisy = eval_ASR(clean, predict, text, asr_model)
        metrics = [wer_clean, wer_noisy]
    else:
        metric1 = batch_pesq(clean, predict, 'wb')
        metric2 = batch_pesq(clean, predict, 'nb')
        metric3 = SI_SDR(clean, predict)
        metric4 = batch_stoi(clean, predict)
        metric5 = LSD(clean, predict)
        metrics = [metric1, metric2, metric3, metric4, metric5]
    return np.stack(metrics, axis=1)

def dot(x, y):
    return torch.sum(x * y, dim=-1, keepdim=True)
def compute_permuted_sisnrs(permuted_pr_batch, t_batch, t_t_diag, eps=10e-8):
    pr_signal_powers = dot(permuted_pr_batch, permuted_pr_batch)
    inner_prod_sq = dot(permuted_pr_batch, t_batch) ** 2
    rho_sq = inner_prod_sq / (pr_signal_powers * t_t_diag + eps)
    return 10 * torch.log10((rho_sq + eps) / (1. - rho_sq + eps))
def sisdr_loss(pr_batch, t_batch, initial_mixtures, eps=1e-8):
    pr_batch = pr_batch - torch.mean(pr_batch, dim=-1, keepdim=True)
    t_batch = t_batch - torch.mean(t_batch, dim=-1, keepdim=True)
    initial_mixtures = initial_mixtures - torch.mean(initial_mixtures, dim=-1, keepdim=True)
    t_t_diag = dot(t_batch, t_batch)

    sisnr = compute_permuted_sisnrs(pr_batch, t_batch, t_t_diag, eps=eps)
    # sisnr = 0.75 * sisnr[:, 0, 0] + 0.25 * sisnr[:, 1, 0]

    #initial_mix = initial_mixtures.repeat(1, 2, 1)
    #base_sisdr = compute_permuted_sisnrs(initial_mix, t_batch, t_t_diag, eps=eps)
    #sisnr -= base_sisdr.mean()
    sisnr = sisnr.mean()
    return -sisnr
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
def train_sudormrf(model, acc, noise, clean, optimizer, device='cuda'):
    # sudormrf only for 8k
    noise = noise[:, ::2]
    clean = clean[:, ::2]
    optimizer.zero_grad()
    noise = noise.unsqueeze(1).to(device=device)
    clean = clean.unsqueeze(1).to(device=device)
    # residual_noise = noise - clean
    # predict = model(noise, acc.to(device=device))
    # loss = sisdr_loss(predict, torch.cat([clean, residual_noise], dim=1), noise)
    # loss = torch.clamp(
    #     loss, min=-30., max=+30.)
    predict = model(noise)
    loss = sisdr_loss(predict, clean, noise)
    loss.backward()
    optimizer.step()
    return loss.item()
def test_sudormrf(model, acc, noise, clean, device='cuda', text=None):
    noise = noise[:, ::2]
    clean = clean[:, ::2]

    noise = noise.unsqueeze(1).to(device=device)
    predict = model(noise)[:, 0, :]
    predict = predict.cpu().numpy()
    clean = clean.numpy()
    return eval(clean, predict, text=text)
def train_voicefilter(model, acc, noise, clean, optimizer, device='cuda'):
    noisy_mag, _, _, _ = stft(noise, 400, 160, 400)
    clean_mag, _, _, _ = stft(clean, 400, 160, 400)
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
    noisy_mag, noisy_phase, _, _ = stft(noise, 400, 160, 400)

    noisy_mag = noisy_mag.to(device=device)
    mask = model(noisy_mag.permute(0, 2, 1), acc.to(device=device)).permute(0, 2, 1)
    predict = noisy_mag * mask

    predict = predict.cpu()
    predict = istft((predict, noisy_phase), 400, 160, 400, input_type="mag_phase").numpy()
    clean = clean.numpy()
    return eval(clean, predict, text=text)
def train_vibvoice(model, acc, noise, clean, optimizer, device='cuda'):
    noisy_mag, _, _, _ = stft(noise, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    optimizer.zero_grad()
    noisy_mag = noisy_mag.to(device=device)
    clean_mag = clean_mag.to(device=device)
    predict, acc = model(noisy_mag, acc)
    loss = Spectral_Loss(torch.squeeze(predict, 1), clean_mag[:, 1:257, 1:])
    loss += 0.1 * F.mse_loss(torch.squeeze(acc, 1), clean_mag[:, :32, 1:])
    loss.backward()
    optimizer.step()
    return loss.item()
def test_vibvoice(model, acc, noise, clean, device='cuda', text=None):
    noisy_mag, noisy_phase, _, _ = stft(noise, 640, 320, 640)
    noisy_mag = noisy_mag.to(device=device)
    predict = model(noisy_mag, acc)
    if isinstance(predict, tuple):
        predict = predict[0]
    predict = predict.cpu()
    predict = F.pad(predict, (1, 0, 1, 321 - 257)).squeeze(1)
    predict = istft((predict, noisy_phase), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy()
    return eval(clean, predict, text=text)
def train_fullsubnet(model, acc, noise, clean, optimizer, device='cuda'):
    optimizer.zero_grad()
    noise = noise.to(device=device)
    clean = clean.to(device=device)

    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noise, 512, 256, 512)
    _, _, clean_real, clean_imag = stft(clean, 512, 256, 512)
    cIRM = build_complex_ideal_ratio_mask(noisy_real.unsqueeze(1), noisy_imag.unsqueeze(1),
                                          clean_real.unsqueeze(1), clean_imag.unsqueeze(1))  # [B, F, T, 2]
    cIRM = cIRM.squeeze(1).permute((0, 3, 1, 2))

    noisy_mag = noisy_mag.unsqueeze(1)
    cRM = model(noisy_mag)
    loss = F.mse_loss(cIRM, cRM)

    loss.backward()
    optimizer.step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
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
    if data:
        noise = noise.numpy()
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
