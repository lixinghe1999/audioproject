import os
import json
import math
import numpy as np
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa
from scipy import interpolate
from audio_zen.acoustics.feature import norm_amplitude, tailor_dB_FS, is_clipped, load_wav, subsample
from torchvision.utils import save_image
from A2net import A2net
from fullsubnet import FullSubNet
import argparse
seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32

rate_mic = 16000
rate_imu = 1600
length = 15
stride = 15
function_pool = '../transfer_function'
#N = len(os.listdir(function_pool))
N = 300

freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(length * rate_mic/(seg_len_mic-overlap_mic)) + 1

sentences = [
    ["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"],
    ["WE", "WANT", "TO", "IMPROVE", "SPEECH", "QUALITY", "IN", "THIS", "PROJECT"],
    ["BUT", "WE", "DON'T", "HAVE", "ENOUGH", "DATA", "TO", "TRAIN", "OUR", "MODEL"],
    ["TRANSFER", "FUNCTION", "CAN", "BE", "A", "GOOD", "HELPER", "TO", "GENERATE", "DATA"]
]

def spectrogram(data, nperseg, noverlap, fs):
    Zxx = signal.stft(data, nperseg=nperseg, noverlap=noverlap, fs=fs, axis=0)[-1]
    if len(Zxx.shape) == 3:
        Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
    Zxx = np.expand_dims(Zxx, 0)
    return Zxx
def read_transfer_function(path):
    npzs = os.listdir(path)
    transfer_function = np.zeros((len(npzs), freq_bin_high))
    variance = np.zeros((len(npzs), freq_bin_high))
    for i in range(len(npzs)):
        npz = np.load(path + '/' + npzs[i])
        transfer_function[i, :] = npz['response']
        variance[i, :] = npz['variance']
    new_transfer_function = np.zeros((N, freq_bin_high))
    new_variance = np.zeros((N, freq_bin_high))
    for j in range(freq_bin_high):
        x = np.linspace(0, 1, len(npzs))
        freq_r = np.sort(transfer_function[:, j])
        freq_v = np.sort(variance[:, j])
        f1 = interpolate.interp1d(x, freq_r)
        f2 = interpolate.interp1d(x, freq_v)
        xnew = np.linspace(0, 1, N)
        new_transfer_function[:, j] = f1(xnew)
        new_variance[:, j] = f2(xnew)
    return new_transfer_function, new_variance

def noise_extraction():
    noise_list = os.listdir('../dataset/noise/')
    index = np.random.randint(0, len(noise_list))
    noise_clip = np.load('../dataset/noise/' + noise_list[index])
    index = np.random.randint(0, noise_clip.shape[1] - time_bin)
    return noise_clip[:, index:index + time_bin]


def synthetic(clean, transfer_function, variance):
    index = np.random.randint(0, N)
    f = transfer_function[index, :]
    f_norm = f / np.max(f)
    v_norm = variance[index, :] / np.max(f)
    response = np.tile(np.expand_dims(f_norm, axis=1), (1, time_bin))
    # for j in range(time_bin):
    #     response[:, j] += np.random.normal(0, v_norm, (freq_bin_high))
    noisy = clean[:, :freq_bin_high, :] * response
    background_noise = noise_extraction()
    noisy += 2 * background_noise
    return noisy

def snr_mix(clean_y, noise_y, snr, target_dB_FS, target_dB_FS_floating_value, rir=None, eps=1e-6):
        """
        Args:
            clean_y: 纯净语音
            noise_y: 噪声
            snr (int): 信噪比
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            rir: room impulse response, None 或 np.array
            eps: eps
        Returns:
            (noisy_y，clean_y)
        """
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]

            clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]

        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y ** 2).mean() ** 0.5

        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value
        )

        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar

        return noisy_y, clean_y
def snr_norm(signals, target_dB_FS, target_dB_FS_floating_value):
        """
        Args:
            signals: list of signal
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
        Returns:
            (noisy_y，clean_y)
        """
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value
        )
        new_signal = []
        for signal in signals:
            signal, _ = norm_amplitude(signal)
            signal, _, _ = tailor_dB_FS(signal, noisy_target_dB_FS)
            new_signal.append(signal)
        return new_signal

class BaseDataset:
    def __init__(self, files=None, pad=False, sample_rate=16000):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.sample_rate = sample_rate
        self.length = length
        self.stride = stride
        for info in self.files:
            _, file_length = info
            if self.length is None:
                examples = 1
            elif file_length < self.length*self.sample_rate:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length * self.sample_rate) / self.stride/self.sample_rate) + 1)
            else:
                examples = (file_length - self.length * self.sample_rate) // (self.stride*self.sample_rate) + 1
            self.num_examples.append(examples)
    def __len__(self):
        return sum(self.num_examples)
    def __getitem__(self, index):
        for info, examples in zip(self.files, self.num_examples):
            file, _ = info
            if index >= examples:
                index -= examples
                continue
            duration = 0
            offset = 0
            if self.length:
                offset = self.stride * index
                duration = self.length
            if file[-3:] == 'txt':
                data = np.loadtxt(file)
                data = data[offset * self.sample_rate: (offset + duration) * self.sample_rate, :]
                data /= 2 ** 14
                b, a = signal.butter(4, 80, 'highpass', fs=self.sample_rate)
                data = signal.filtfilt(b, a, data, axis=0)
                data = np.clip(data, -0.05, 0.05)
            else:
                data, _ = librosa.load(file, offset=offset, duration=duration, sr=rate_mic)

            return data, file
class NoisyCleanSet:
    def __init__(self, json_paths, text=False, person=None, simulation=False, ratio=1, snr=(0, 20)):
        '''
        :param json_paths: speech (clean), noisy/ added noise, IMU (optional)
        :param text: whether output the text, only apply to Sentences
        :param person: person we want to involve
        :param simulation: whether the noise is simulation
        :param ratio: ratio of the data we use
        :param snr: SNR range of the synthetic dataset
        '''
        self.dataset = []
        self.ratio = ratio
        sr = [16000, 16000, 1600]
        for i, path in enumerate(json_paths):
            with open(path, 'r') as f:
                data = json.load(f)
            if person is not None and isinstance(data, dict):
                tmp = []
                for p in person:
                    if ratio > 0:
                        tmp += data[p][:int(len(data[p]) * self.ratio)]
                    else:
                        tmp += data[p][int(len(data[p]) * self.ratio):]
                data = tmp
            else:
                if ratio > 0:
                    data = data[:int(len(data) * self.ratio)]
                else:
                    data = data[int(len(data) * self.ratio):]
            self.dataset.append(BaseDataset(data, sample_rate=sr[i]))
        if len(json_paths) == 2:
            self.augmentation = True
            transfer_function, variance = read_transfer_function(function_pool)
            self.variance = variance
            self.transfer_function = transfer_function
        else:
            self.augmentation = False
        self.simulation = simulation
        self.text = text
        self.length = len(self.dataset[1])
        self.snr_list = np.arange(snr[0], snr[1], 1)
    def __getitem__(self, index):
        clean, file = self.dataset[0][index]
        if self.simulation:
            noise, _ = self.dataset[1][np.random.randint(0, self.length)]
            snr = np.random.choice(self.snr_list)
            noise, clean = snr_mix(clean, noise, snr, -25, 10, rir=None, eps=1e-6)
        else:
            noise, _ = self.dataset[1][index]
            noise, clean = snr_norm([clean, noise], -25, 10)
        noise = spectrogram(noise, seg_len_mic, overlap_mic, rate_mic)
        clean = spectrogram(clean, seg_len_mic, overlap_mic, rate_mic)
        if self.augmentation:
            imu = synthetic(np.abs(clean), self.transfer_function, self.variance)
        else:
            imu, _ = self.dataset[2][index]
            imu = spectrogram(imu, seg_len_imu, overlap_imu, rate_imu)
        noise = noise[:, 1:8 * freq_bin_high, :-1]
        clean = clean[:, 1:8 * freq_bin_high, :-1]
        imu = imu[:, 1:, :-1]
        if self.text:
            return file, imu, noise, clean
        else:
            return imu, noise, clean
    def __len__(self):
        return len(self.dataset[0])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-pre train, 1-main benchmark, 2-mirco benchmark')
    args = parser.parse_args()
    def model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        #print('model size: {:.3f}MB'.format(size_all_mb))
        return size_all_mb
    if args.mode == 0:
        # check data
        #dataset_train = NoisyCleanSet(['json/train.json', 'json/dev.json'], simulation=True, ratio=1)
        dataset_train = NoisyCleanSet(['json/position_gt.json', 'json/position_gt.json','json/position_imu.json'],
                                      simulation=True, person=['headphone'])
        print(len(dataset_train))
        loader = Data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=False)
        for step, (x, noise, y) in enumerate(loader):
            print(x.shape, noise.shape, y.shape)

            x = x[0, 0].numpy()
            noise = np.abs(noise[0, 0].numpy())
            y = np.abs(y[0, 0].numpy())
            fig, axs = plt.subplots(3, 1)
            axs[0].imshow(x, aspect='auto')
            axs[1].imshow(np.abs(noise[:freq_bin_high, :]), aspect='auto')
            axs[2].imshow(np.abs(y[:freq_bin_high, :]), aspect='auto')
            plt.show()
    elif args.mode == 1:
        # save different positions correlation with audio
        # dataset = NoisyCleanSet(['json/train_gt.json', 'json/train_gt.json', 'json/train_imu'], person=['hou'],
        #                               simulation=True, ratio=1)
        for position in ['glasses', 'vr-up', 'vr-down', 'headphone-inside', 'headphone-outside', 'cheek', 'temple', 'back', 'nose']:
            dataset = NoisyCleanSet(['json/position_gt.json', 'json/position_gt.json', 'json/position_imu.json'],
                                          simulation=True, person=[position])
            loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
            Corr = []
            for step, (x, noise, y) in enumerate(loader):
                x = x[0, 0].numpy()
                noise = torch.abs(noise[0, 0]).numpy()
                y = torch.abs(y[0, 0]).numpy()
                # wave_x = np.mean(x, axis=0)
                # wave_y = np.mean(y, axis=0)
                _, wave_x = signal.istft(x, fs=rate_imu, nperseg=seg_len_imu, noverlap=overlap_imu)
                _, wave_y = signal.istft(y, fs=rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)
                corr = np.corrcoef(wave_x, wave_y[::10])[0, 1]
                if corr != corr:
                    continue
                else:
                    Corr.append(corr)
            print(position, np.mean(Corr))
    else:
        # model check 1. FullSubNet 2. A2Net
        model1 = Model(
            num_freqs=264,
            look_ahead=2,
            sequence_model="LSTM",
            fb_num_neighbors=0,
            sb_num_neighbors=15,
            fb_output_activate_function="ReLU",
            sb_output_activate_function=False,
            fb_model_hidden_size=512,
            sb_model_hidden_size=384,
            norm_type="offline_laplace_norm",
            num_groups_in_drop_band=2,
            weight_init=False,
        )
        model2 = A2net()
        size1 = model_size(model1)
        size2 = model_size(model2)
        print(size1, size2)
        with torch.no_grad():
            dataset_train = NoisyCleanSet(['json/noise_train_gt.json', 'json/noise_train_wav.json', 'json/noise_train_imu.json'], simulation=True, person=['he'])
            loader = Data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=False)
            for step, (x, noise, y) in enumerate(loader):
                x, noise, y = x.to(dtype=torch.float), torch.abs(noise).to(dtype=torch.float), torch.abs(y).to(dtype=torch.float)
                # x1 = model1(noise)
                x1, x2 = model2(x, noise)
                print(x1.shape)
