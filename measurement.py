import matplotlib.pyplot as plt
import os
import torch
import imuplot
from DL.evaluation import wer
import os
from pesq import pesq
import torchaudio
import micplot
import soundfile as sf
import librosa
import numpy as np
import scipy.signal as signal
from matplotlib.ticker import MaxNLocator
from speechbrain.pretrained import EncoderDecoderASR
import argparse
# seg_len_mic = 640
# overlap_mic = 320
# seg_len_imu = 64
# overlap_imu = 32

seg_len_mic = 2560
overlap_mic = 2240
seg_len_imu = 256
overlap_imu = 224

rate_mic = 16000
rate_imu = 1600
T = 5
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_limit = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(T*rate_mic/(seg_len_mic-overlap_mic)) + 1

def synchronize(x, x_t, y_t):
    x_t = np.sum(x_t, axis=0)
    y_t = np.sum(y_t, axis=0)
    corr = signal.correlate(y_t, x_t, 'full')
    shift = np.argmax(corr) - np.shape(x_t)
    x = np.roll(x, shift, axis=1)

    return x
def data_extract(path, files_mic1, files_mic2, files_imu1, files_imu2, i):
    wave_1 = librosa.load(os.path.join(path, files_mic1[i]), sr=rate_mic)[0]
    wave_2 = librosa.load(os.path.join(path, files_mic2[i]), sr=rate_mic)[0]

    b, a = signal.butter(4, 100, 'highpass', fs=16000)
    wave_1 = signal.filtfilt(b, a, wave_1)
    wave_2 = signal.filtfilt(b, a, wave_2)

    Zxx1 = signal.stft(wave_1, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1]
    Zxx2 = signal.stft(wave_2, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1]

    data1, imu1 = imuplot.read_data(os.path.join(path, files_imu1[i]), seg_len_imu, overlap_imu, rate_imu)
    data2, imu2 = imuplot.read_data(os.path.join(path, files_imu2[i]), seg_len_imu, overlap_imu, rate_imu)
    return Zxx1, Zxx2, imu1, imu2
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=1, required=False,
                        help='mode of processing, 0-WER, 1-spectrogram compare, 2-low, high frequency')
    args = parser.parse_args()
    if args.mode == 0:
        ground_truth = {'s1': ["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"],
                        's2': ["WE", "WANT", "TO", "IMPROVE", "SPEECH", "QUALITY", "IN", "THIS", "PROJECT"],
                        's3': ["BUT", "WE", "DON'T", "HAVE", "ENOUGH", "DATA", "TO", "TRAIN", "OUR", "MODEL"],
                        's4': ["TRANSFER", "FUNCTION", "CAN", "BE", "A", "GOOD", "HELPER", "TO", "GENERATE", "DATA"]}
        sentences = ['s1', 's2', 's3', 's4']
        plt_WER = []
        for earphone in ['galaxy', 'freebud', 'airpod', 'logitech']:
            WER_one = []
            for s in sentences:
                gt = ground_truth[s]
                path = os.path.join('exp7', earphone, s)
                if earphone == 'logitech':
                    files = os.listdir(path)
                else:
                    files = os.listdir(path)[1::2]
                N = len(files)
                asr_model = EncoderDecoderASR.from_hparams(source="pretrained_models/asr-transformer-transformerlm-librispeech",
                                                           savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
                                                           run_opts={"device": "cuda"})
                WER = []
                for i in range(N):
                    text = asr_model.transcribe_file(os.path.join(path, files[i])).split()
                    print(wer(gt, text), wer(text, gt))
                    WER.append(wer(gt, text))
                WER_one.append(sum(WER) / len(WER))
            plt_WER.append(WER_one)
        x = np.arange(len(sentences))
        width = 0.15
        plt.bar(x, plt_WER[0], width=width, label='galaxy', fc='b')
        plt.bar(x + width, plt_WER[1], width=width, label='freebud', fc='r')
        plt.bar(x + 2 * width, plt_WER[2], width=width, label='airpod', fc='g')
        plt.bar(x + 3 * width, plt_WER[3], width=width, label='logitech', fc='y')
        plt.xticks(x + width * 1.5, sentences)
        plt.legend()
        plt.title('Word Error Rate')
        plt.ylabel('WER/%')
        plt.savefig('wer_pesq.eps', dpi=300)

    elif args.mode == 1:
        start = 50
        end = 100
        sentences = ['s1', 's2', 's3', 's4']
        for s in sentences:
            path = os.path.join('exp7/hou', s, 'mobile')
            files = os.listdir(path)
            N = int(len(files) / 4)
            files_imu1 = files[:N]
            files_imu2 = files[N:2 * N]
            files_mic1 = files[2 * N:3 * N]
            files_mic2 = files[3 * N:]
            files_mic2.sort(key=lambda x: int(x[4:-4]))
            for i in range(N):
                Zxx1, Zxx2, imu1, imu2 = data_extract(path, files_mic1, files_mic2, files_imu1, files_imu2, i)
                imu1 = synchronize(imu1, imu1, np.abs(Zxx1[freq_bin_low:freq_bin_high, :]))

                fig, axs = plt.subplots(1, 3, sharey=True, figsize=(5, 3))
                fig.subplots_adjust(top=0.97, right=0.97, bottom=0.14, wspace=0.1, hspace=0)

                for ax in axs:
                    ax.xaxis.label.set_color('white')
                    ax.xaxis.set_label_coords(0.5, 0.08)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                axs[0].imshow(np.abs(Zxx1[freq_bin_low:freq_bin_high, start:end]), extent=[0, 1, 800, 100], aspect='auto')
                axs[1].imshow(np.abs(Zxx2[freq_bin_low:freq_bin_high, start:end]), extent=[0, 1, 800, 100], aspect='auto')
                axs[2].imshow(imu1[freq_bin_low:, start:end], extent=[0, 1, 800, 100], aspect='auto')
                axs[0].set_xlabel('Mic')
                axs[1].set_xlabel('Mic')
                axs[2].set_xlabel('Acc')
                fig.text(0.45, 0.022, 'Time (Second)', va='center')
                fig.text(0.01, 0.52, 'Frequency (Hz)', va='center', rotation='vertical')
                plt.savefig('compare.eps', dpi=300)
                plt.show()
    else:
        start = 50
        end = 100
        sentences = ['s1', 's2', 's3', 's4']
        for s in sentences:
            path = os.path.join('exp7/hou', s, 'mobile')
            files = os.listdir(path)
            N = int(len(files) / 4)
            files_imu1 = files[:N]
            files_imu2 = files[N:2 * N]
            files_mic1 = files[2 * N:3 * N]
            files_mic2 = files[3 * N:]
            files_mic2.sort(key=lambda x: int(x[4:-4]))
            for i in range(N):
                Zxx1, Zxx2, imu1, imu2 = data_extract(path, files_mic1, files_mic2, files_imu1, files_imu2, i)
                imu1 = synchronize(imu1, imu1, np.abs(Zxx1[freq_bin_low:freq_bin_high, :]))

                fig, axs = plt.subplots(1, 2, figsize=(5, 3))
                fig.subplots_adjust(top=0.97, right=0.97, bottom=0.14, wspace=0.2, hspace=0)

                for ax in axs:
                    ax.xaxis.label.set_color('white')
                    ax.xaxis.set_label_coords(0.5, 0.08)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                axs[0].imshow(np.abs(Zxx1[freq_bin_high: 2 * freq_bin_high, start:end]), extent=[0, 1, 2 * 800, 800], aspect='auto')
                axs[1].imshow(imu1[freq_bin_low:, start:end], extent=[0, 1, 800, 100], aspect='auto')
                axs[0].set_xlabel('Mic')
                axs[1].set_xlabel('Acc')
                fig.text(0.45, 0.022, 'Time (Second)', va='center')
                fig.text(0.01, 0.52, 'Frequency (Hz)', va='center', rotation='vertical')
                plt.savefig('compare.eps', dpi=300)
                plt.show()


