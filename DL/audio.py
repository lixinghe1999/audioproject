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
def harmonic_bandwidth_extension(x, y):
    values = []
    for b in range(len(x)):
        gt = y[b, 0].numpy()
        gt = np.pad(gt, ((freq_bin_low, 0), (0, 0)))
        gt_phase = np.angle(gt)
        gt_abs = np.abs(gt)

        x_LF = x[b, 0].numpy()
        x_LF = np.exp(1j * gt_phase[freq_bin_low:freq_bin_high, :]) * x_LF
        x_LF = np.pad(x_LF, ((freq_bin_low, 0), (0, 0)))
        x_LF[int(freq_bin_high/2):, :] = 0

        time_stretch = librosa.phase_vocoder(x_LF, rate=0.5, hop_length=seg_len_imu-overlap_imu)
        time_series = librosa.istft(time_stretch, hop_length=seg_len_imu - overlap_imu, win_length=seg_len_imu)
        time_series = time_series[::2]
        new_spectrogram = librosa.stft(time_series, n_fft=seg_len_imu, hop_length=seg_len_imu - overlap_imu)
        x_LF[int(freq_bin_high / 2):, :] = new_spectrogram[int(freq_bin_high / 2):, :]


        _, gt_audio = signal.istft(gt, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)
        gt[:freq_bin_high, :] = x_LF
        # recover_audio = librosa.griffinlim(gt, n_iter=0, hop_length=seg_len_mic-overlap_mic, win_length=seg_len_mic )
        _, recover_audio = signal.istft(gt, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

        recover_audio = recover_audio * 0.0037
        gt_audio = gt_audio * 0.0037
        value = pesq(16000, recover_audio, gt_audio, 'nb')
        print(value)
        if value > 0:
            values.append(value)
        # sf.write('result.wav', recover_audio, 16000)
        # sf.write('ground_truth.wav', gt_audio, 16000)
        # fig, axs = plt.subplots(2)
        # axs[0].imshow(np.abs(x_LF), aspect='auto')
        # axs[1].imshow(gt_abs[:freq_bin_high, :], aspect='auto')
        # plt.show()
    return values
def cal_pesq_mel(x, y, model, device):
    x_abs = x.to(dtype=torch.float, device=device)
    predict = model(x_abs).cpu().numpy()
    x_abs = x_abs.cpu()
    y = y.numpy()
    values = []
    for b in range(len(x)):
        gt = y[b, 0]
        predict_y = predict[b, 0]

        gt = librosa.feature.inverse.mel_to_stft(gt, sr=rate_mic, n_fft=seg_len_mic, power=1, fmin=100, fmax=800)
        gt_audio = librosa.griffinlim(gt, n_iter=12, hop_length=seg_len_mic - overlap_mic, win_length=seg_len_mic)
        gt_abs = gt.copy()

        predict_y = librosa.feature.inverse.mel_to_stft(predict_y, sr=rate_imu, n_fft=seg_len_imu, power=1, fmin=100)
        gt[:freq_bin_high, :] = predict_y
        recover_audio = librosa.griffinlim(gt, n_iter=12,  hop_length=seg_len_mic - overlap_mic, win_length=seg_len_mic)

        recover_audio = recover_audio * 0.73
        gt_audio = gt_audio * 0.73
        value = pesq(16000, recover_audio, gt_audio, 'nb')
        print(value)
        if value > 0:
            values.append(value)
        sf.write('result.wav', recover_audio, 16000)
        sf.write('ground_truth.wav', gt_audio, 16000)
        fig, axs = plt.subplots(3)
        axs[0].imshow(x_abs[b, 0, :, :], aspect='auto')
        axs[1].imshow(predict_y, aspect='auto')
        axs[2].imshow(gt_abs[freq_bin_low:freq_bin_high, :], aspect='auto')
        plt.show()
    return values
def cal_pesq(x, y, model, device):
    # Note that x can be magnitude, y need to have phase
    x_abs = x.to(dtype=torch.float, device=device)
    predict = model(x_abs).cpu().numpy()
    x_abs = x_abs.cpu()
    y = y.numpy()
    values = []
    for b in range(len(x)):
        gt = y[b, 0]
        predict_y = predict[b, 0]

        gt = np.pad(gt, ((freq_bin_low, 0), (0, 0)))
        #gt[freq_bin_high:, 0] = 0
        _, gt_audio = signal.istft(gt, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)


        gt_phase = np.angle(gt)
        gt_abs = np.abs(gt)
        recover = np.exp(1j * gt_phase[freq_bin_low:freq_bin_high, :]) * predict_y
        gt[freq_bin_low:freq_bin_high, :] = recover
        #recover_audio = librosa.griffinlim(gt, n_iter=0, hop_length=seg_len_mic-overlap_mic, win_length=seg_len_mic )
        _, recover_audio = signal.istft(gt, rate_mic, nperseg=seg_len_mic, noverlap=overlap_mic)

        recover_audio = recover_audio * 0.0037
        gt_audio = gt_audio * 0.0037
        value = pesq(16000, recover_audio, gt_audio, 'nb')
        print(value)
        if value > 0:
            values.append(value)
        # sf.write('result.wav', recover_audio, 16000)
        # sf.write('ground_truth.wav', gt_audio, 16000)
        # fig, axs = plt.subplots(3)
        # axs[0].imshow(x_abs[b, 0, :, :], aspect='auto')
        # axs[1].imshow(predict_y, aspect='auto')
        # axs[2].imshow(gt_abs[freq_bin_low:freq_bin_high, :], aspect='auto')
        # plt.show()
    return values
if __name__ == "__main__":
    # load IMU audio ground truth
    BATCH_SIZE = 16
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = UNet(1, 1).to(device)

    model.load_state_dict(torch.load("checkpoint_4_0.011230765171919246.pth"))
    #model.load_state_dict(torch.load("checkpoints/finetuneIMUs_2.75.pth"))
    dataset = IMUSPEECHSet('clean_imuexp6.json', 'clean_wavexp6.json', phase=True, full=True, minmax=(0.0132, 0.0037))
    #dataset = IMUSPEECHSet('noise_imuexp6.json', 'noise_wavexp6.json-', ground_truth=False, phase=True, minmax=(0.012, 0.002))
    length = len(dataset)
    train_size, validate_size = int(0.8 * length), length - int(0.8 * length)
    dataset, test = torch.utils.data.random_split(dataset, [train_size, validate_size], torch.Generator().manual_seed(0))

    test_loader = Data.DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False)
    PESQ = []
    with torch.no_grad():
        for x, y in test_loader:
            #values = harmonic_bandwidth_extension(x, y)
            values = cal_pesq(x, y, model, device)
            PESQ += values
    plt.plot(PESQ)
    print(np.mean(PESQ))
    plt.show()
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
