import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc

rc('text', usetex=True)
plt.rcParams.update({'font.size': 10})
import imuplot
from DL.evaluation import wer
import os

import librosa
import numpy as np
import scipy.signal as signal
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
def data_extract(path, files_mic1, files_mic2, files_imu1, files_imu2):
    wave_1 = librosa.load(os.path.join(path, files_mic1), sr=rate_mic)[0]
    wave_2 = librosa.load(os.path.join(path, files_mic2), sr=rate_mic)[0]

    # b, a = signal.butter(4, 100, 'highpass', fs=16000)
    # wave_1 = signal.filtfilt(b, a, wave_1)
    # wave_2 = signal.filtfilt(b, a, wave_2)

    Zxx1 = np.abs(signal.stft(wave_1, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1])
    Zxx2 = np.abs(signal.stft(wave_2, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1])

    data1, imu1 = imuplot.read_data(os.path.join(path, files_imu1), seg_len_imu, overlap_imu, rate_imu, filter=True)
    data2, imu2 = imuplot.read_data(os.path.join(path, files_imu2), seg_len_imu, overlap_imu, rate_imu, filter=True)
    imu1 = np.pad(imu1, ((0, freq_bin_high), (0, 0)))
    imu2 = np.pad(imu2, ((0, freq_bin_high), (0, 0)))
    return Zxx1, Zxx2, imu1, imu2
def draw_2(Zxx, imu, start, stop, n, vmax):
    fig, axs = plt.subplots(1, 2, figsize=(4, 2))
    plt.subplots_adjust(left=0.14, bottom=0.12, right=1, top=0.92, wspace=-0.03, hspace=0)
    spectrogram1 = Zxx[: 2 * freq_bin_high, int(start * 50): int(stop * 50)]
    spectrogram2 = imu[: 2 * freq_bin_high, int(start * 50): int(stop * 50)]

    rect = patches.Rectangle((1.05, 200), 0.7, 550, linewidth=1, edgecolor='w', facecolor='none')
    axs[0].add_patch(rect)
    axs[0].locator_params(axis='x', nbins=1)
    axs[0].set_yticks([0, 100, 400, 800, 1600])
    axs[0].axline((0, 100), (2, 100), color='w')
    im1 = axs[0].imshow(spectrogram1, extent=[0, stop - start, 0, 1600],
                        aspect='auto', origin='lower', vmin=0, vmax=vmax[0])
    cb1 = fig.colorbar(im1, ticks=[], ax=axs[0], aspect=50)
    cb1.ax.text(2.5, 0.05, '0', transform=cb1.ax.transAxes, va='top', ha='center')
    cb1.ax.text(1.1, 1, str(vmax[0]), transform=cb1.ax.transAxes, va='bottom', ha='center')


    axs[1].locator_params(axis='x', nbins=1)
    axs[1].set_yticks([])
    axs[1].axline((0, 100), (2, 100), color='w')
    im2 = axs[1].imshow(spectrogram2, extent=[0, stop - start, 0, 1600],
                        aspect='auto', origin='lower', vmin=0, vmax=vmax[1])
    cb2 = fig.colorbar(im2, ticks=[], ax=axs[1], aspect=50)
    cb2.ax.text(2.5, 0.05, '0', transform=cb2.ax.transAxes, va='top', ha='center')
    cb2.ax.text(1.1, 1, str(vmax[1]), transform=cb2.ax.transAxes, va='bottom', ha='center')

    fig.text(0.2, 0.95, 'Microphone', va='center')
    fig.text(0.65, 0.95, 'Acceleration', va='center')
    fig.text(0.2, 0.04, 'Time(Sec)', va='center')
    fig.text(0.65, 0.04, 'Time(Sec)', va='center')
    fig.text(0.01, 0.50, 'Frequency(Hz)', va='center', rotation='vertical')
    #plt.savefig(n, dpi=600)
    plt.show()

def draw_4(spectrogram1, spectrogram2, index, vmax):


    #axs[2*index].locator_params(axis='x', nbins=1)
    #axs[2*index].set_yticks([0, 100, 400, 800, 1600])
    axs[2 * index].set_xticks([])
    axs[2 * index].set_title('Mic')
    axs[2*index].axline((0, 100), (2, 100), color='w')
    im1 = axs[2*index].imshow(spectrogram1, extent=[0, 2, 0, 1600],
                        aspect='auto', origin='lower', vmin=0, vmax=vmax[0])
    # cb1 = fig.colorbar(im1, ticks=[], ax=axs[0], aspect=50)
    # cb1.ax.text(2.5, 0.05, '0', transform=cb1.ax.transAxes, va='top', ha='center')
    # cb1.ax.text(1.1, 1, str(vmax[0]), transform=cb1.ax.transAxes, va='bottom', ha='center')

    #axs[2*index + 1].locator_params(axis='x', nbins=1)
    #axs[2*index + 1].set_yticks([])
    axs[2 * index + 1].set_xticks([])
    axs[2 * index + 1].set_title('Acc')
    axs[2*index + 1].axline((0, 100), (2, 100), color='w')
    im2 = axs[2*index + 1].imshow(spectrogram2, extent=[0, 2, 0, 1600],
                        aspect='auto', origin='lower', vmin=0, vmax=vmax[1])
    # cb2 = fig.colorbar(im2, ticks=[], ax=axs[1], aspect=50)
    # cb2.ax.text(2.5, 0.05, '0', transform=cb2.ax.transAxes, va='top', ha='center')
    # cb2.ax.text(1.1, 1, str(vmax[1]), transform=cb2.ax.transAxes, va='bottom', ha='center')

    # fig.text(0.2, 0.95, 'Microphone', va='center')
    # fig.text(0.65, 0.95, 'Acceleration', va='center')
    # fig.text(0.2, 0.04, 'Time(Sec)', va='center')
    # fig.text(0.65, 0.04, 'Time(Sec)', va='center')
    # fig.text(0.01, 0.50, 'Frequency(Hz)', va='center', rotation='vertical')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-WER, 1-spectrogram compare, 2-low, high frequency')
    args = parser.parse_args()
    if args.mode == 0:
        functions = os.listdir('transfer_function')
        collect1 = {}
        collect2 = {}
        for f in functions:
            r = np.load('transfer_function/' + f)['response']
            v = np.load('transfer_function/' + f)['variance']
            id = int(f.split('_')[0])
            if id not in collect1:
                collect1[id] = r
                collect2[id] = v
            else:
                collect1[id] = np.column_stack([collect1[id], r])
                collect2[id] = np.column_stack([collect2[id], r])

        fig, axs = plt.subplots(1, figsize=(5, 3))
        for i in range(2):
            response = np.mean(collect1[i], axis=1)
            variance = np.mean(collect2[i], axis=1)
            #plt.plot(response)
            plt.errorbar(range(33), response, yerr=variance/3, fmt='-o', label='volunteer'+str(i))
        plt.xticks([0, 7, 15, 23, 31], [0, 200, 400, 600, 800])
        plt.legend()

        fig.text(0.4, 0.02, 'Frequency(Hz)', va='center')
        fig.text(0.02, 0.575, '${S_Acc}/{S_Mic}$', va='center', rotation='vertical')
        plt.savefig('transfer_variance.pdf', dpi=600)
        plt.show()

    elif args.mode == 1:
        path = os.path.join('dataset/measurement/')
        files = os.listdir(path)
        N = int(len(files) / 4)
        files_imu1 = files[:N]
        files_imu2 = files[N:2 * N]
        files_mic1 = files[2 * N:3 * N]
        files_mic2 = files[3 * N:]
        files_mic2.sort(key=lambda x: int(x[4:-4]))
        crop = [[0.5, 2.5], [1.5, 3.5], [0.5, 2.5], [1, 3]]
        name = ['clean', 'noise', 'move', 'notalk']
        #vmax = [[[0.05, 0.0008], [0.02, 0.02]], [[0.05, 0.0008], [0.02, 0.02]], [[0.05, 0.0008], [0.02, 0.02]], [[0.05, 0.0008], [0.02, 0.02]]]
        vmax = [[0.05, 0.02], [0.05, 0.02], [0.05, 0.02], [0.05, 0.02]]
        select = [1, 5, 6, 13]
        for i in [0, 1, 2, 3]:
            v = vmax[i]
            start = crop[i][0]
            stop = crop[i][1]
            n = name[i]
            i = select[i]

            Zxx1, Zxx2, imu1, imu2 = data_extract(path, files_mic1[i], files_mic2[i], files_imu1[i], files_imu2[i])
            draw_2(Zxx2, imu1, start, stop, n + '.pdf', v)
    else:
        path = os.path.join('dataset/measurement/')
        files = os.listdir(path)
        N = int(len(files) / 4)
        files_imu1 = files[:N]
        files_imu2 = files[N:2 * N]
        files_mic1 = files[2 * N:3 * N]
        files_mic2 = files[3 * N:]
        files_mic2.sort(key=lambda x: int(x[4:-4]))

        crop = [[0.5, 2.5], [1.5, 3.5]]
        vmax = [[0.05, 0.02], [0.05, 0.02]]
        select = [1, 5]
        fig, axs = plt.subplots(1, 4, figsize=(4, 2), sharey=True)
        plt.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.88, wspace=0.05, hspace=0.05)
        for i in [0, 1]:
            index = select[i]
            Zxx1, Zxx2, imu1, imu2 = data_extract(path, files_mic1[index], files_mic2[index], files_imu1[index], files_imu2[index])
            v = vmax[i]
            start = crop[i][0]
            stop = crop[i][1]
            spectrogram1 = Zxx2[: 2 * freq_bin_high, int(start * 50): int(stop * 50)]
            spectrogram2 = imu1[: 2 * freq_bin_high, int(start * 50): int(stop * 50)]
            draw_4(spectrogram1, spectrogram2, i, v)

        axs[0].set_yticks([0, 100, 400, 800, 1600])
        rect = patches.Rectangle((1.05, 200), 0.7, 550, linewidth=1, edgecolor='w', facecolor='none')
        axs[2].add_patch(rect)
        fig.text(0.5, 0.05, 'Time')
        plt.savefig('measurement.pdf', dpi=600)
        plt.show()

        # crop = [[0.5, 2.5], [1.5, 3.5], [0.5, 2.5], [1, 3]]
        # name = ['clean', 'noise', 'move', 'notalk']
        # # vmax = [[[0.05, 0.0008], [0.02, 0.02]], [[0.05, 0.0008], [0.02, 0.02]], [[0.05, 0.0008], [0.02, 0.02]], [[0.05, 0.0008], [0.02, 0.02]]]
        # vmax = [[0.05, 0.02], [0.05, 0.02], [0.05, 0.02], [0.05, 0.02]]
        # for i in [1]:
        #     select = [1, 5, 6, 13]
        #     v = vmax[i]
        #     start = crop[i][0]
        #     stop = crop[i][1]
        #     n = name[i]
        #     i = select[i]
        #
        #     Zxx1, Zxx2, imu1, imu2 = data_extract(path, files_mic1[i], files_mic2[i], files_imu1[i], files_imu2[i])
        #     draw(Zxx2, imu1, start, stop, n + '.pdf', v)

