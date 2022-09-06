'''
simulation includes:
1) pygsound: geometric-based
2) pyroomacoustics: image-based
'''
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile

fs = 16000
Lg_t = 0.100                # filter size in seconds
Lg = np.ceil(Lg_t*fs)       # in samples
# specify signal and noise source
fs, signal = wavfile.read("example.wav")
fs, noise = wavfile.read("noise.wav")  # may spit out a warning when reading but it's alright!

# Create 4x6 shoebox room with source and interferer and simulate
room_bf = pra.ShoeBox([4,6], fs=fs, max_order=12)
source = np.array([2, 4])
interferer = np.array([4, 2])
room_bf.add_source(source, delay=0., signal=signal)
room_bf.add_source(interferer, delay=0., signal=noise[:len(signal)])

# Create geometry equivalent to Amazon Echo
center = [2, 2]; radius = 0.1
fft_len = 512
echo = pra.circular_2D_array(center=center, M=6, phi0=0, radius=radius)
echo = np.concatenate((echo, np.array(center, ndmin=2).T), axis=1)
mics = pra.Beamformer(echo, room_bf.fs, N=fft_len, Lg=Lg)
room_bf.add_microphone_array(mics)

# Compute DAS weights
mics.rake_delay_and_sum_weights(room_bf.sources[0][:1])

room_bf.compute_rir()
room_bf.simulate()
signal_das = mics.process(FD=False)
signal_das = signal_das/ np.max(signal_das)
signal_noise = room_bf.mic_array.signals[1:, :].T
print(signal_noise.shape)
signal_noise = signal_noise/ np.max(signal_noise)
wavfile.write('noisy.wav', fs, signal_noise.astype(np.float32))
wavfile.write('DAS.wav', fs, signal_das.astype(np.float32))


# fig, axs = plt.subplots(3)
# axs[0].plot(signal)
# axs[1].plot(room_bf.mic_array.signals[-1,:])
# axs[2].plot(signal_das)
# plt.show()
