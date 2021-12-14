# utilize all the result before, generate audio and then use evaluation
import scipy.signal as signal
import soundfile as sf
from pesq import pesq
import torch
import torch.utils.data as Data
from data import NoisyCleanSet, IMUSPEECHSet, transfer_function_generator, read_transfer_function, weighted_loss
from unet import UNet
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
time_bin = int(segment * rate_mic/(seg_len_mic-overlap_mic)) + 1
if __name__ == "__main__":
    # load IMU audio ground truth
    BATCH_SIZE = 1
    model = UNet(1, 1)
    transfer_function, variance, noise = read_transfer_function('transfer_function')
    transfer_function = transfer_function_generator(transfer_function)
    variance = transfer_function_generator(variance)

    model.load_state_dict(torch.load("checkpoints/train_0_0.02572406893459094.pth"))
    #dataset = NoisyCleanSet(transfer_function, variance, noise, 'speech100.json', alpha=(6, 0.012, 0.0583))
    dataset = IMUSPEECHSet('imuexp6.json', 'wavexp6.json', with_path=False, phase=True, minmax=(0.012, 0.002))
    test_loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            y_abs = torch.abs(y)
            phase = torch.angle(y)
            x_abs, y_abs = x.to(dtype=torch.float), y_abs.to(dtype=torch.float)

            x_t = np.sum(x[0, 0, :, :].numpy(), axis=0)
            y_t = np.sum(y_abs[0, 0, :, :].numpy(), axis=0)
            corr = signal.correlate(y_t, x_t, 'full')
            shift = np.argmax(corr) - len(x_t)
            print(shift)
            x_abs = torch.roll(x_abs, shift, dims=3)

            predict = model(x_abs)
            print(weighted_loss(x_abs, y_abs, 0.02), weighted_loss(predict, y_abs, 0.02))


            #     shift = np.argmax(corr) - len(x_t)
            recover = np.exp(1j * phase.numpy()) * predict.numpy() * 0.002
            truth = np.exp(1j * phase.numpy()) * y_abs.numpy() * 0.002

            spectrogram = np.zeros((int(seg_len_mic / 2) + 1, time_bin), dtype='complex64')
            spectrogram[freq_bin_low:freq_bin_high, :] = recover
            #audio = librosa.griffinlim(spectrogram, 10, seg_len_mic-overlap_mic, seg_len_mic)
            _, audio = signal.istft(spectrogram, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

            spectrogram[freq_bin_low:freq_bin_high, :] = truth
            _, out = signal.istft(spectrogram, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)
            sf.write('result.wav', audio, 16000)
            sf.write('ground_truth.wav', out, 16000)

            fig, axs = plt.subplots(3)
            axs[0].imshow(x_abs[0, 0, :, :], aspect='auto')
            axs[1].imshow(y_abs[0, 0, :, :], aspect='auto')
            axs[2].imshow(predict[0, 0, :, :], aspect='auto')
            plt.show()
            #spectrogram = Zxx
            #spectrogram[:freq_bin_low, :] = Zxx[:freq_bin_low, :]
            # fig, axs = plt.subplots(2)
            # axs[0].imshow(np.abs(Zxx[freq_bin_low:200, :]), aspect='auto')
            # axs[1].imshow(np.abs(spectrogram[freq_bin_low:200, :]), aspect='auto')
            # plt.show()

            # fig, axs = plt.subplots(2)
            # axs[0].imshow(y_abs[0, 0, :, :], aspect='auto')
            # axs[1].imshow(Zxx[freq_bin_low:freq_bin_high, :], aspect='auto')
            # plt.show()

