# utilize all the result before, generate audio and then use evaluation
import scipy.signal as signal
import soundfile as sf
from pesq import pesq
import torch
import torch.nn as nn
import torch.utils.data as Data
from data import NoisyCleanSet, IMUSPEECHSet
from A2net import A2net

from torch.nn.utils.rnn import pad_sequence
from evaluation import wer
from speechbrain.pretrained import EncoderDecoderASR, SpectralMaskEnhancement
from speechbrain.pretrained import SepformerSeparation as separator
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torchaudio
import os
import pydub


seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32
rate_mic = 16000
rate_imu = 1600
length = 5
stride = 4

freq_bin_high = 8 * (int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1)
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(length * rate_mic/(seg_len_mic-overlap_mic)) + 1
freq_bin_limit = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
Loss = nn.L1Loss()
sentences = [["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"],
                    ["WE", "WANT", "TO", "IMPROVE", "SPEECH", "QUALITY", "IN", "THIS", "PROJECT"],
                    ["BUT", "WE", "DON'T", "HAVE", "ENOUGH", "DATA", "TO", "TRAIN", "OUR", "MODEL"],
                    ["TRANSFER", "FUNCTION", "CAN", "BE", "A", "GOOD", "HELPER", "TO", "GENERATE", "DATA"]]

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
    text = [t.split() for t in text]
    return text
def towav(name, predict):
    predict = np.pad(predict, ((0, int(seg_len_mic / 2) + 1 - freq_bin_high), (0, 0)))
    _, recover_audio = signal.istft(predict, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)
    recover_audio = 1/np.max(recover_audio)
    sf.write(name, recover_audio, 16000)
    return recover_audio
def measurement_vibvoice(x, noise, y, model, asr_model, device, num_sentence):
    # Note that x can be magnitude, y need to have phase
    x_abs, noise_abs, y_abs = x.to(device=device, dtype=torch.float), torch.abs(noise).to(device=device, dtype=torch.float), torch.abs(y)
    predict, _ = model(x_abs, noise_abs)
    predict = predict.cpu().numpy()

    y = y.numpy()
    noise = noise.numpy()
    values = []
    error = []
    for b in range(len(x)):
        gt = y[b, 0]
        n = noise[b, 0]
        predict = predict[b, 0]
        phase = np.angle(n)

        gt_audio = towav('ground_truth.wav', gt)
        recover_audio = towav('extra.wav', np.exp(1j * phase[:freq_bin_high, :]) * predict)

        value = pesq(16000, recover_audio, gt_audio, 'nb')
        values.append(value)
        audio_files = ['extra.wav']
        text1 = batch_ASR(audio_files, asr_model)
        text = sentences[num_sentence-1]
        error_list = wer(text, text1)
        error.append(error_list)
    return values, error

def measurement_baseline(x, noise, y, baseline_model1, baseline_model2, asr_model, num_sentence):
    # Note that x can be magnitude, y need to have phase
    y = y.numpy()
    noise = noise.numpy()
    values = []
    error = []
    for b in range(len(x)):
        gt = y[b, 0]
        n = noise[b, 0]
        ori_audio = towav('ori_audio.wav', n)
        gt_audio = towav('ground_truth.wav', gt)

        # baseline1 - enhancement
        noisy = baseline_model1.load_audio("ori_audio.wav").unsqueeze(0)
        enhanced = baseline_model1.enhance_batch(noisy, lengths=torch.tensor([1.])).cpu()
        torchaudio.save('baseline.wav', enhanced, 16000)

        # baseline2 - separation
        est_sources = baseline_model2.separate_file(path='ori_audio.wav')
        est_source1, est_source2 = est_sources[:, :, 0].detach().cpu(), est_sources[:, :, 1].detach().cpu()
        torchaudio.save("source1.wav", est_source1, 8000)
        torchaudio.save("source2.wav", est_source2, 8000)

        value1 = max(pesq(16000, est_source1.numpy()[0], gt_audio, 'nb'), pesq(16000, est_source2.numpy()[0], gt_audio, 'nb'))
        value2 = pesq(16000, enhanced.numpy()[0], gt_audio, 'nb')
        value3 = pesq(16000, ori_audio, gt_audio, 'nb')

        values.append([value1, value2, value3])
        audio_files = ['source1.wav', 'source2.wav', 'baseline.wav', 'ori_audio.wav', 'ground_truth.wav']
        text1, text2, text3, text4, text5 = batch_ASR(audio_files, asr_model)
        text = sentences[num_sentence-1]
        error_list = [min(wer(text, text1), wer(text, text2)), wer(text, text3), wer(text, text4), wer(text, text5)]
        error.append(error_list)
    return values, error

def save_audio(x, noise, y, model, enhance_model, device, count):
    # Note that x can be magnitude, y need to have phase
    x_abs, noise_abs, y_abs = x.to(device=device, dtype=torch.float), torch.abs(noise).to(device=device, dtype=torch.float), torch.abs(y)
    predict1, _ = model(x_abs, noise_abs)
    predict1 = predict1.cpu().numpy()

    y = y.numpy()
    noise = noise.numpy()
    for b in range(len(x)):
        gt = y[b, 0]
        n = noise[b, 0]
        phase = np.angle(n)
        predict1 = predict1[b, 0]

        gt_audio = towav('ground_truth.wav', gt)
        ori_audio = towav('ori_audio.wav', n)
        recover_audio = towav('extra.wav', np.exp(1j * phase[:freq_bin_high, :]) * predict1)


        noisy = enhance_model.load_audio('ori_audio.wav').unsqueeze(0)
        enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.])).cpu()
        torchaudio.save('baseline.wav', enhanced, 16000)

        value1 = pesq(16000, recover_audio, gt_audio, 'nb')
        value2 = pesq(16000, enhanced.numpy()[0], gt_audio, 'nb')

        if (value1 - value2) > 0:
            print(value1, value2)
            sf.write('result/' + str(count) + '_ground_truth.wav', gt_audio * 0.1, 16000)
            sf.write('result/' + str(count) + '_ori_audio.wav', ori_audio * 0.1, 16000)
            sf.write('result/' + str(count) + '_model.wav', recover_audio * 0.1, 16000)
            sf.write('result/' + str(count) + '_baseline.wav', enhanced.numpy()[0], 16000)
            count += 1
    return count

def data_para(target, flag):
    if flag == 0:
        file = open("noise_paras.pkl", "rb")
        paras = pickle.load(file)
        dataset = IMUSPEECHSet('noise_imuexp7.json', 'noise_gtexp7.json', 'noise_wavexp7.json', person=[target],
                               simulate=False, phase=True, minmax=paras[target])
    elif flag == 1:
        file = open("clean_paras.pkl", "rb")
        paras = pickle.load(file)
        dataset = IMUSPEECHSet('clean_imuexp7.json', 'clean_wavexp7.json', 'clean_wavexp7.json', person=[target],
                               phase=True, minmax=paras[target])
    elif flag == 2:
        file = open("mobile_paras.pkl", "rb")
        paras = pickle.load(file)
        dataset = IMUSPEECHSet('mobile_imuexp7.json', 'mobile_wavexp7.json', 'mobile_wavexp7.json', person=[target],
                               phase=True, minmax=paras[target])
    else:
        file = open("field_paras.pkl", "rb")
        paras = pickle.load(file)
        dataset = IMUSPEECHSet('field_imuexp7.json', 'field_gtexp7.json', 'field_wavexp7.json', person=[target],
                               simulate=False, phase=True, minmax=paras[target])
    return dataset

def baseline(target, flag):
    BATCH_SIZE = 1
    dataset = data_para(target, flag)
    asr_model = EncoderDecoderASR.from_hparams(source="../pretrained_models/asr-transformer-transformerlm-librispeech",
                                               savedir="../pretrained_models/asr-transformer-transformerlm-librispeech",
                                               run_opts={"device": "cuda"})
    baseline_model1 = SpectralMaskEnhancement.from_hparams(source="../pretrained_models/metricgan-plus-voicebank",
                                                         savedir="../pretrained_models/metricgan-plus-voicebank",
                                                         run_opts={"device": "cuda"})
    baseline_model2 = separator.from_hparams(source="../speechbrain/sepformer-whamr", savedir='../pretrained_models/sepformer-whamr',
                                   run_opts={"device": "cuda"})
    test_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    PESQ = []
    WER = []
    with torch.no_grad():
        for num_sentence, x, noise, y in test_loader:
            values, error = measurement_baseline(x, noise, y, baseline_model1, baseline_model2, asr_model, num_sentence)
            print(error)
            PESQ += values
            WER += error
    return PESQ, WER

def vibvoice(ckpt, target, flag):
    BATCH_SIZE = 1
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = A2net().to(device)
    model.load_state_dict(ckpt)
    dataset = data_para(target, flag)
    asr_model = EncoderDecoderASR.from_hparams(source="../pretrained_models/asr-transformer-transformerlm-librispeech",
                                               savedir="../pretrained_models/asr-transformer-transformerlm-librispeech",
                                               run_opts={"device": "cuda"})
    test_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    PESQ = []
    WER = []
    with torch.no_grad():
        for num_sentence, x, noise, y in test_loader:
            values, error = measurement_vibvoice(x, noise, y, model, asr_model, device, num_sentence)
            PESQ += values
            WER += error
    return PESQ, WER

if __name__ == "__main__":
    # 0-noise, 1-clean, 2-mobile, 3-field
    #candidate = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
    candidate = ["he", "hou"]
    folder = 'checkpoint/baseline/clean/'
    for target in candidate:
        PESQ, WER = baseline(target, 1)
        mean_PESQ = np.mean(PESQ, axis=0)
        mean_WER = np.mean(WER, axis=0)
        gt = mean_WER[-1]
        np.savez(folder + target + '_' + str(gt) + '.npz', PESQ=PESQ, WER=WER)

    # for f in os.listdir('checkpoint/5min'):
    #     if f.split('_')[0] == target and f[-3:] == 'pth':
    #         test(torch.load('checkpoint/5min/' + f), target, 2)


