# utilize all the result before, generate audio and then use evaluation
import scipy.signal as signal
import soundfile as sf
from pesq import pesq
import torch
import torch.utils.data as Data
from data import NoisyCleanSet, IMUSPEECHSet, transfer_function_generator, read_transfer_function, weighted_loss
from unet import UNet, TinyUNet
import librosa
import numpy as np
import matplotlib.pyplot as plt

seg_len_mic = 2560
overlap_mic = 2240
rate_mic = 16000
seg_len_imu = 256
overlap_imu = 224
rate_imu = 1600
segment = 4
stride = 1
N = 50
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(segment * rate_mic / (seg_len_mic - overlap_mic)) + 1


def spec2audio(spec):
    spectrogram = np.zeros((int(seg_len_mic / 2) + 1, time_bin), dtype='complex64')
    spectrogram[freq_bin_low:freq_bin_high, :] = spec
    _, audio = signal.istft(spectrogram, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)
    return audio
def cal_pesq(x, y, model):
    # Note that x can be magnitude, y need to have phase
    y_abs = torch.abs(y)
    phase = torch.angle(y)

    _, gt_audio = signal.istft(y, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)
    x_abs = x.to(dtype=torch.float)
    predict = model(x_abs)
    recover = torch.exp(1j * phase[0, 0, freq_bin_low:freq_bin_high, :]) * predict
    y[:, :, freq_bin_low:freq_bin_high, :] = recover
    _, recover_audio = signal.istft(y, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

    recover_audio = recover_audio[0, 0] * 0.012
    gt_audio = gt_audio[0, 0] * 0.012

    value = pesq(16000, recover_audio, gt_audio, 'nb')
    print(value)

    sf.write('result.wav', recover_audio, 16000)
    sf.write('ground_truth.wav', gt_audio, 16000)
    fig, axs = plt.subplots(3)
    axs[0].imshow(x_abs[0, 0, :, :], aspect='auto')
    axs[1].imshow(predict[0, 0, :, :], aspect='auto')
    axs[2].imshow(y_abs[0, 0, freq_bin_low:freq_bin_high, :], aspect='auto')
    plt.show()
    return value
if __name__ == "__main__":
    # load IMU audio ground truth
    BATCH_SIZE = 1
    model = UNet(1, 1)
    #model = TinyUNet(1, 1)
    transfer_function, variance, noise = read_transfer_function('transfer_function')
    transfer_function = transfer_function_generator(transfer_function)
    variance = transfer_function_generator(variance)

    model.load_state_dict(torch.load("checkpoints/finetuneIMUs_0.019480812113865147.pth"))
    #dataset = NoisyCleanSet(transfer_function, variance, noise, 'speech100.json', alpha=(6, 0.012, 0.0583))
    #dataset = NoisyCleanSet(transfer_function, variance, noise, 'devclean.json', alpha=(6, 0.012, 0.0583))
    dataset = IMUSPEECHSet('clean_imuexp6.json', 'clean_wavexp6.json', phase=True, full=True, minmax=(0.012, 0.002))
    #dataset = IMUSPEECHSet('noise_imuexp6.json', 'noise_wavexp6.json-', ground_truth=False, phase=True, minmax=(0.012, 0.002))
    length = len(dataset)
    train_size, validate_size = int(0.8 * length), length - int(0.8 * length)
    dataset, test = torch.utils.data.random_split(dataset, [train_size, validate_size], torch.Generator().manual_seed(0))

    test_loader = Data.DataLoader(dataset=test, batch_size=1, shuffle=False)
    PESQ = []
    with torch.no_grad():
        for x, y in test_loader:
            value = cal_pesq(x, y, model)
            PESQ.append(value)
    # plt.plot(PESQ)
    # print(np.mean(PESQ))
    # plt.show()
            # y_abs = torch.abs(y)
            # phase = torch.angle(y)
            # x_abs, y_abs = x.to(dtype=torch.float), y_abs.to(dtype=torch.float)
            # predict = model(x_abs)
            #
            # original = np.exp(1j * phase.numpy()) * x_abs.numpy() * 0.012
            # recover = np.exp(1j * phase.numpy()) * predict.numpy() * 0.002
            # y = np.exp(1j * phase.numpy()) * y_abs.numpy() * 0.002
            #
            # original = spec2audio(x_abs.numpy() * 0.002)
            # recover = spec2audio(recover)
            # y = spec2audio(y)
            # #
            # # truth, sr = librosa.load(z[0], sr=rate_mic)
            # # b, a = signal.butter(4, 100, 'highpass', fs=rate_mic)
            # # truth = signal.filtfilt(b, a, truth)
            # # corr = signal.correlate(truth, y, 'valid')
            # # shift = np.argmax(corr)
            # # truth = truth[shift: shift + len(y)]
            # # out = np.abs(signal.stft(truth, nperseg=seg_len_mic, noverlap=overlap_mic, fs=sr)[-1])
            #
