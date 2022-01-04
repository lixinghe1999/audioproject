import matplotlib.pyplot as plt
import os
import torch
import imuplot
from evaluation import wer
import os
from pesq import pesq
import torchaudio
import micplot
import soundfile as sf
import librosa
import numpy as np
import scipy.signal as signal
from speechbrain.pretrained import EncoderDecoderASR
from DL.unet import UNet
seg_len_mic = 2560
overlap_mic = 2240
rate_mic = 16000
seg_len_imu = 256
overlap_imu = 224
rate_imu = 1600
T = 5
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(T*rate_mic/(seg_len_mic-overlap_mic)) + 1

def synchronize(x, x_t, y_t):
    x_t = np.sum(x_t, axis=0)
    y_t = np.sum(y_t, axis=0)
    corr = signal.correlate(y_t, x_t, 'full')
    shift = np.argmax(corr) - np.shape(x_t)
    x = np.roll(x, shift, axis=1)

    return x
def data_extract(i):
    wave_1 = librosa.load(os.path.join(path, files_mic1[i]), sr=rate_mic)[0]
    wave_2 = librosa.load(os.path.join(path, files_mic2[i]), sr=rate_mic)[0]
    b, a = signal.butter(4, 100, 'highpass', fs=16000)
    wave_1 = signal.filtfilt(b, a, wave_1)
    wave_2 = signal.filtfilt(b, a, wave_2)

    Zxx1 = signal.stft(wave_1, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1]
    Zxx2 = signal.stft(wave_2, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1]

    data1, imu1, t_imu1 = imuplot.read_data(os.path.join(path, files_imu1[i]), seg_len_imu, overlap_imu, rate_imu)
    data2, imu2, t_imu2 = imuplot.read_data(os.path.join(path, files_imu2[i]), seg_len_imu, overlap_imu, rate_imu)
    return Zxx1, Zxx2, imu1[freq_bin_low:, :], imu2[freq_bin_low:, :]
def inference(x, y, gt):
    model = UNet(1, 1)
    model.load_state_dict(torch.load("DL/checkpoint_5_0.010252494036563134.pth"))
    with torch.no_grad():
        x = np.expand_dims(x, (0, 1))
        x_abs = torch.from_numpy(x).to(dtype=torch.float)
        predict = model(x_abs/0.0137).numpy()
        x_abs = x_abs.numpy()
        gt_abs = np.abs(gt)
        # gt[freq_bin_high:, :] = 0
        _, gt_audio = signal.istft(gt, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

        recover = np.exp(1j * np.angle(gt)[freq_bin_low:freq_bin_high, :]) * predict
        gt[freq_bin_low:freq_bin_high, :] = recover
        _, recover_audio = signal.istft(gt, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

        recover_audio = recover_audio
        gt_audio = gt_audio
        value = pesq(16000, recover_audio, gt_audio, 'nb')
        print(value)
        sf.write('result.wav', recover_audio, 16000)
        sf.write('ground_truth.wav', gt_audio, 16000)
        # fig, axs = plt.subplots(3)
        # axs[0].imshow(x_abs[0, 0], aspect='auto')
        # axs[1].imshow(np.abs(predict[0, 0]), aspect='auto')
        # axs[2].imshow(gt_abs[freq_bin_low:freq_bin_high, :], aspect='auto')
        # plt.show()
    return value
if __name__ == "__main__":
    # location: Experiment/subject/position/classification
    # S1
    # conversation = range(6)
    # music = range(6, 12)
    # clean = range(12, 19)
    # mask = range(19, 25)
    # S2
    # conversation = range(6)
    # music = range(6, 12)
    # clean = range(12, 17)
    # mask = range(17, 23)

    ground_truth = {'s1': ["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"],
                    's2': ["WE", "WANT", "TO", "IMPROVE", "SPEECH", "QUALITY", "IN", "THIS", "PROJECT"]}
    sentences = ['s1', 's2']
    plt_good = []
    plt_bad = []
    plt_recover = []
    PESQ = []

    for s in sentences:
        gt = ground_truth[s]
        path = os.path.join('exp7\he_calibrated', s)
        files = os.listdir(path)

        N = int(len(files) / 4)
        files_imu1 = files[:N]
        files_imu2 = files[N:2 * N]
        files_mic1 = files[2 * N:3 * N]
        files_mic2 = files[3 * N:]
        files_mic2.sort(key=lambda x: int(x[4:-4]))
        asr_model = EncoderDecoderASR.from_hparams(source="pretrained_models/asr-transformer-transformerlm-librispeech",
                                                   savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
                                                   run_opts={"device": "cuda"})
        good = []
        bad = []
        recover = []
        audio_files = []
        for i in range(12):
            text1 = asr_model.transcribe_file(os.path.join(path, files_mic1[i])).split()
            text2 = asr_model.transcribe_file(os.path.join(path, files_mic2[i])).split()
            good.append(wer(text1, gt))
            bad.append(wer(text2, gt))
            audio_1 = librosa.load(os.path.join(path, files_mic1[i]), sr=rate_mic)[0]
            audio_2 = librosa.load(os.path.join(path, files_mic2[i]), sr=rate_mic)[0]
            PESQ.append(pesq(16000, audio_1, audio_2, 'nb'))

            Zxx1, Zxx2, imu1, imu2 = data_extract(i)
            imu1 = synchronize(imu1, imu1, np.abs(Zxx1[freq_bin_low:freq_bin_high,:]))
            inference(imu1, Zxx2, Zxx1)
            text3 = asr_model.transcribe_file('result.wav').split()
            print(wer(text3, gt))
            text4 = asr_model.transcribe_file('groundtruth.wav').split()
            print(wer(text4, gt))
        plt_good.append(np.mean(good))
        plt_bad.append(np.mean(bad))
        plt_recover.append(np.mean(recover))

    fig, axs = plt.subplots(1, 2)
    x = np.arange(len(sentences))
    width = 0.3
    axs[0].bar(x, plt_good, width=width, label='Logitech', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    axs[0].bar(x+ width, plt_bad, width=width, label='Airpod', fc='r')
    axs[0].bar(x + 2 * width, plt_recover, width=width, label='Recover', fc='b')
    axs[0].set_xticks(x + width)
    axs[0].set_xticklabels(sentences)
    axs[0].legend()
    axs[0].set_title('Word Error Rate')

    axs[1].plot(PESQ)
    axs[1].set_title('PESQ')
    plt.show()


