# utilize all the result before, generate audio and then use evaluation
import scipy.signal as signal
import soundfile as sf
from pesq import pesq
import torch
import torch.nn as nn
import torch.utils.data as Data
from data import NoisyCleanSet, IMUSPEECHSet, read_transfer_function
from unet import UNet, ExtendNet
from A2net import A2net
from A2netU import A2netU
from A2netcomplex import A2net_m, A2net_p
from evaluation import wer
from torch.nn.utils.rnn import pad_sequence
import pyrubberband
from speechbrain.pretrained import EncoderDecoderASR, SpectralMaskEnhancement
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torchaudio


seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32
rate_mic = 16000
rate_imu = 1600
length = 5
stride = 1
freq_bin_high = 8 * (int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1)
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(length * rate_mic/(seg_len_mic-overlap_mic)) + 1
freq_bin_limit = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
Loss = nn.L1Loss()
sentences = [["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"],
                    ["WE", "WANT", "TO", "IMPROVE", "SPEECH", "QUALITY", "IN", "THIS", "PROJECT"],
                    ["BUT", "WE", "DON'T", "HAVE", "ENOUGH", "DATA", "TO", "TRAIN", "OUR", "MODEL"],
                    ["TRANSFER", "FUNCTION", "CAN", "BE", "A", "GOOD", "HELPER", "TO", "GENERATE", "DATA"]]
def groundtruth(text):
    error_rate = 1000
    for gt in sentences:
        r = wer(gt, text)
        if r < error_rate:
            error_rate = r
            gt_text = gt
    return error_rate, gt_text

def HBE(stft, r):
    stft = librosa.phase_vocoder(stft, 1/r, hop_length=seg_len_mic-overlap_mic)
    # plt.imshow(np.abs(stft[:, :]), aspect='auto')
    # plt.show()
    _, t = signal.istft(stft, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)
    #t = librosa.griffinlim(stft, n_iter=5, hop_length=seg_len_mic - overlap_mic, win_length=seg_len_mic)
    t = t[::r]
    stft = librosa.stft(t, n_fft=seg_len_mic, hop_length=seg_len_mic - overlap_mic)

    return stft[(r-1) * freq_bin_limit:r * freq_bin_limit, :]


def harmonic_bandwidth_extension(x, y):
    values = []
    for b in range(len(x)):
        gt = y[b, 0].numpy()
        gt_phase = np.angle(gt)

        x_LF = x[b, 0].numpy()
        x_LF = np.exp(1j * gt_phase[:freq_bin_limit, :]) * x_LF
        x_LF = np.pad(x_LF, ((0, int(seg_len_mic/2) + 1 - freq_bin_limit), (0, 0)))

        _, gt_audio = signal.istft(gt, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)
        for r in [2]:
            x_LF[(r-1) * freq_bin_limit:r * freq_bin_limit, :] = 0.01 * HBE(x_LF, r)
            # recover_audio = librosa.griffinlim(gt, n_iter=0, hop_length=seg_len_mic-overlap_mic, win_length=seg_len_mic )
        _, recover_audio = signal.istft(x_LF, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)
        recover_audio = recover_audio * 0.0037
        gt_audio = gt_audio * 0.0037
        value = pesq(16000, recover_audio, gt_audio, 'nb')
        print(value)
        if value > 0:
            values.append(value)
        # sf.write('result.wav', recover_audio, 16000)
        # sf.write('ground_truth.wav', gt_audio, 16000)
        fig, axs = plt.subplots(2)
        axs[0].imshow(np.abs(x_LF[:2*freq_bin_limit, :]), aspect='auto')
        axs[1].imshow(np.abs(gt[:, :]), aspect='auto')
        plt.show()
    return values
def batch_ASR(audio_files, asr_model):
    sigs = []
    lens = []
    for audio_file in audio_files:
        snt, fs = torchaudio.load(audio_file)
        sigs.append(snt.squeeze())
        lens.append(snt.shape[1])

    batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)

    lens = torch.Tensor(lens) / batch.shape[1]
    text = asr_model.transcribe_batch(batch, lens)[0]
    text1, text2, text3, text4, text5, text6 = text
    text1 = text1.split()
    text2 = text2.split()
    text3 = text3.split()
    text4 = text4.split()
    text5 = text5.split()
    text6 = text6.split()
    return text1, text2, text3, text4, text5, text6
def measurement(x, noise, y,  model, extra_model, audio_model, enhance_model, asr_model):
    # Note that x can be magnitude, y need to have phase
    x_abs, noise_abs, y_abs = x.to(dtype=torch.float), torch.abs(noise).to(dtype=torch.float), torch.abs(y)
    #x_abs, noise_abs = x.to(dtype=torch.float), noise.to(dtype=torch.float)
    predict1 = model(x_abs, noise_abs)
    predict2, _ = extra_model(x_abs, noise_abs)
    predict3 = audio_model(x_abs, noise_abs)
    #print(Loss(noise_abs, y_abs).item(), Loss(predict1, y_abs).item())

    predict1 = predict1.numpy()
    predict2 = predict2.numpy()
    predict3 = predict3.numpy()
    y = y.numpy()
    noise = noise.numpy()
    values = []
    error = []
    for b in range(len(x)):
        gt = y[b, 0]
        n = noise[b, 0]
        predict1 = predict1[b, 0]
        predict2 = predict2[b, 0]
        predict3 = predict3[b, 0]

        phase = np.angle(n)
        gt_abs = np.abs(gt)
        n_abs = np.abs(n)

        n = np.pad(n, ((0, int(seg_len_mic / 2) + 1 - freq_bin_high), (0, 0)))
        _, ori_audio = signal.istft(n, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

        gt = np.pad(gt, ((0, int(seg_len_mic / 2) + 1 - freq_bin_high), (0, 0)))
        _, gt_audio = signal.istft(gt, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

        predict1 = np.exp(1j * phase[:freq_bin_high, :]) * predict1
        predict1 = np.pad(predict1, ((0, int(seg_len_mic / 2) + 1 - freq_bin_high), (0, 0)))
        _, recover_audio1 = signal.istft(predict1, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

        predict2 = np.exp(1j * phase[:freq_bin_high, :]) * predict2
        predict2 = np.pad(predict2, ((0, int(seg_len_mic / 2) + 1 - freq_bin_high), (0, 0)))
        _, recover_audio2 = signal.istft(predict2, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

        predict3 = np.exp(1j * phase[:freq_bin_high, :]) * predict3
        predict3 = np.pad(predict3, ((0, int(seg_len_mic / 2) + 1 - freq_bin_high), (0, 0)))
        _, recover_audio3 = signal.istft(predict3, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

        recover_audio1 = recover_audio1 * 0.1
        recover_audio2 = recover_audio2 * 0.1
        recover_audio3 = recover_audio3 * 0.1
        gt_audio = gt_audio * 0.1
        ori_audio = ori_audio * 0.1

        sf.write('vanilla.wav', recover_audio1, 16000)
        sf.write('extra.wav', recover_audio2, 16000)
        sf.write('single.wav', recover_audio3, 16000)
        sf.write('ground_truth.wav', gt_audio, 16000)
        sf.write('ori_truth.wav', ori_audio, 16000)

        # baseline
        noisy = enhance_model.load_audio("ori_truth.wav").unsqueeze(0)
        enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.])).cpu()
        torchaudio.save('baseline.wav', enhanced, 16000)

        value1 = pesq(16000, recover_audio1, gt_audio, 'nb')
        value2 = pesq(16000, recover_audio2, gt_audio, 'nb')
        value3 = pesq(16000, recover_audio3, gt_audio, 'nb')
        value4 = pesq(16000, enhanced.numpy()[0], gt_audio, 'nb')
        value5 = pesq(16000, ori_audio, gt_audio, 'nb')

        if value5 < 4:
            # only collect when there is noise
            print(value1, value2, value3, value4, value5)
            values.append([value1, value2, value3, value4, value5])
            audio_files = ['vanilla.wav', 'extra.wav', 'single.wav', 'baseline.wav', 'ori_truth.wav', 'ground_truth.wav']
            text1, text2, text3, text4, text5, text6 = batch_ASR(audio_files, asr_model)
            error_rate, text = groundtruth(text6)
            error_list = [error_rate, wer(text, text1), wer(text, text2), wer(text, text3), wer(text, text4), wer(text, text5)]
            error.append(error_list)
            print(error_list)

            # text5 = asr_model.transcribe_file('ori_truth.wav').split()
            # text6 = asr_model.transcribe_file('ground_truth.wav').split()
            # error_rate, text = groundtruth(text6)
            # error_list = [error_rate, 0, 0, 0, wer(text, text5)]
            # error.append(error_list)
            # print(error_list)

            # fig, axs = plt.subplots(3)
            # axs[0].imshow(n_abs[:4 * freq_bin_limit, :], aspect='auto')
            # axs[1].imshow(np.abs(predict1[:4 * freq_bin_limit, :]), aspect='auto')
            # axs[2].imshow(gt_abs[:4 * freq_bin_limit, :], aspect='auto')
            # plt.show()
    return values, error
if __name__ == "__main__":
    BATCH_SIZE = 1

    # model = A2net_m()
    # model.load_state_dict(torch.load("vanilla/he_0.0015024848786803584.pth"))
    # extra_model = A2net_m(extra_supervision=True)
    # extra_model.load_state_dict(torch.load("imu_extra/he_0.0014843073828766744.pth"))
    # audio_model = A2net_m(audio=True)
    # audio_model.load_state_dict(torch.load("audio/he_0.003179994955038031.pth"))

    model = A2net_m()
    model.load_state_dict(torch.load("vanilla/hou_0.0015416461663941541.pth"))
    extra_model = A2net_m(extra_supervision=True)
    extra_model.load_state_dict(torch.load("imu_extra/hou_0.001500579536271592.pth"))
    audio_model = A2net_m(audio=True)
    audio_model.load_state_dict(torch.load("audio/hou_0.0035039723850786688.pth"))

    # model = A2net_m()
    # model.load_state_dict(torch.load("vanilla/shuai_0.0015271187643520535.pth"))
    # extra_model = A2net_m(extra_supervision=True)
    # extra_model.load_state_dict(torch.load("imu_extra/shuai_0.0014795155419657627.pth"))
    # audio_model = A2net_m(audio=True)
    # audio_model.load_state_dict(torch.load("audio/shuai_0.0018135160906240344.pth"))

    # model = A2net_m()
    # model.load_state_dict(torch.load("vanilla/shi_0.0019689142626399796.pth"))
    # extra_model = A2net_m(extra_supervision=True)
    # extra_model.load_state_dict(torch.load("imu_extra/shi_0.0020512066238249343.pth"))
    # audio_model = A2net_m(audio=True)
    # audio_model.load_state_dict(torch.load("audio/shi_0.001915329974144697.pth"))

    asr_model = EncoderDecoderASR.from_hparams(source="../pretrained_models/asr-transformer-transformerlm-librispeech",
                                               savedir="../pretrained_models/asr-transformer-transformerlm-librispeech",
                                               run_opts={"device": "cuda"})
    enhance_model = SpectralMaskEnhancement.from_hparams(source="../pretrained_models/metricgan-plus-voicebank",
                                                         savedir="../pretrained_models/metricgan-plus-voicebank",
                                                         run_opts={"device": "cuda"})

    clean_paras = {'he': (0.0235, 0.1, 0.1), 'hou': (0.029, 0.13, 0.13)}
    noise_paras = {'he': (0.0225, 0.138, 0.125), 'hou': (0.026, 0.14, 0.19), 'shi':(0.0116, 0.12, 0.043), 'shuai': (0.0087, 0.053, 0.051)}
    for c in ['hou']:
        #dataset = IMUSPEECHSet('clean_imuexp7.json', 'clean_wavexp7.json', 'clean_wavexp7.json', person=c, phase=True, minmax=clean_paras[c])
        dataset = IMUSPEECHSet('noise_imuexp7.json', 'noise_gtexp7.json', 'noise_wavexp7.json', person=c, simulate=False, phase=True, minmax=noise_paras[c])

        test_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

        PESQ = []
        WER = []
        with torch.no_grad():
            for x, noise, y in test_loader:
                #values = harmonic_bandwidth_extension(x, y)
                values, error = measurement(x, noise, y, model, extra_model, audio_model, enhance_model, asr_model)
                PESQ += values
                WER += error
        PESQ = np.mean(PESQ, axis=0)
        WER = np.mean(WER, axis=0)
        print('processing finished')
        print(PESQ)
        print(WER)

