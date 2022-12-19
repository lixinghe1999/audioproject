import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile

c = 343.  # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size
freq_range = [300, 3500]

def DOA(signals, mics, azimuth, num_src=2):
    X = pra.transform.stft.analysis(signals, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])

    #algo_names = ['SRP', 'MUSIC', 'FRIDA', 'TOPS']
    algo_names = ['MUSIC']
    spatial_resp = dict()

    # loop through algos
    for algo_name in algo_names:
        # Construct the new DOA object
        # the max_four parameter is necessary for FRIDA only
        doa = pra.doa.algorithms[algo_name](mics, fs, nfft, c=c, num_src=num_src, max_four=4)

        # this call here perform localization on the frames in X
        doa.locate_sources(X, freq_range=freq_range)

        # store spatial response
        if algo_name == 'FRIDA':
            spatial_resp[algo_name] = np.abs(doa._gen_dirty_img())
        else:
            spatial_resp[algo_name] = doa.grid.values

        # normalize
        min_val = spatial_resp[algo_name].min()
        max_val = spatial_resp[algo_name].max()
        spatial_resp[algo_name] = (spatial_resp[algo_name] - min_val) / (max_val - min_val)
    # plotting param
    base = 1.
    height = 10.
    true_col = [0, 0, 0]

    # loop through algos
    phi_plt = doa.grid.azimuth
    for algo_name in algo_names:
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        c_phi_plt = np.r_[phi_plt, phi_plt[0]]
        c_dirty_img = np.r_[spatial_resp[algo_name], spatial_resp[algo_name][0]]
        ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=3,
                alpha=0.55, linestyle='-',
                label="spatial spectrum")
        plt.title(algo_name)

        # plot true loc
        for angle in azimuth:
            ax.plot([angle, angle], [base, base + height], linewidth=3, linestyle='--',
                    color=true_col, alpha=0.6)
        # K = len(azimuth)
        # ax.scatter(azimuth, base + height * np.ones(K), c=np.tile(true_col,
        #                                                           (K, 1)), s=500, alpha=0.75, marker='*',
        #            linewidths=0,
        #            label='true locations')
        ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
        ax.xaxis.set_label_coords(0.5, -0.11)
        ax.set_yticks(np.linspace(0, 1, 2))
        ax.xaxis.grid(visible=True, color=[0.3, 0.3, 0.3], linestyle=':')
        ax.yaxis.grid(visible=True, color=[0.3, 0.3, 0.3], linestyle='--')
        ax.set_ylim([0, 1.05 * (base + height)])
    plt.show()

if __name__ == "__main__":
    wav_files = ['example.wav', 'noise.wav']
    azimuth = np.array([150., 270.]) / 180. * np.pi
    distance = 2.  # meters
    snr_db = 5.    # signal-to-noise ratio
    sigma2 = 10**(-snr_db / 10) / (4. * np.pi * distance)**2

    # Create an anechoic room
    room_dim = np.r_[10.,10.]
    aroom = pra.ShoeBox(room_dim, fs=fs, max_order=1, sigma2_awgn=sigma2)

    echo = pra.circular_2D_array(center=room_dim/2, M=6, phi0=0, radius=0.1)
    echo = np.concatenate((echo, np.array(room_dim/2, ndmin=2).T), axis=1)
    # echo = pra.linear_2D_array(center=room_dim/2, M=6, phi=50, d=0.2)
    # echo = np.concatenate((echo, np.array(room_dim/2, ndmin=2).T), axis=1)
    aroom.add_microphone_array(pra.MicrophoneArray(echo, aroom.fs))
    # Add sources of 1 second duration
    rng = np.random.RandomState(23)
    duration_samples = int(fs)

    for wav, ang in zip(wav_files, azimuth):
        source_location = room_dim / 2 + distance * np.r_[np.cos(ang), np.sin(ang)]
        fs, source_signal = wavfile.read(wav)
        aroom.add_source(source_location, signal=source_signal)

    # Run the simulation
    aroom.simulate()
    DOA(aroom.mic_array.signals.T, echo, azimuth)

