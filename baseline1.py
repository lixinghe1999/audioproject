import micplot
import noisereduce as nr
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
# method to denoise only by speech
if __name__ == "__main__":
    fig, axs = plt.subplots(2,1)
    data, _, _ = micplot.get_wav('test.wav')
    seg_len_mic = 2048
    overlap_mic = 1024
    rate_mic = 16000

    freq_bin = int(1600 / rate_mic * (int(seg_len_mic / 2) + 1))
    reduced_noise = nr.reduce_noise(y = data, sr=rate_mic, thresh_n_mult_nonstationary=1, stationary=False, freq_mask_smooth_hz=100,
                                    n_std_thresh_stationary=1, n_fft=372)
    #reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0, stationary=False)
    micplot.save_wav(reduced_noise, 'reduced_test.wav')
    f, t, Zxx = signal.stft(data, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)
    axs[1].imshow(np.abs(Zxx)[:freq_bin*2, :], aspect='auto')
    axs[0].plot(reduced_noise)
    plt.show()