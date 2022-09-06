# concatanate audio samples to make them look long enough
from scipy.io import wavfile
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

from mir_eval.separation import bss_eval_sources

# Callback function to monitor the convergence of the algorithm
def convergence_callback(Y):
    global SDR, SIR
    y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
    y = y[L - hop:, :].T
    m = np.minimum(y.shape[1], ref.shape[1])
    sdr, sir, sar, perm = bss_eval_sources(ref[:,:m], y[:,:m])
    SDR.append(sdr)
    SIR.append(sir)

wav_files = ['long_example1.wav', 'long_example2.wav']

signals = [wavfile.read(f)[-1] for f in wav_files]

room_dim = [8, 9]

# source locations and delays
locations = [[2.5,3], [2.5, 6]]
delays = [1., 0.]

# create an anechoic room with sources and mics
room = pra.ShoeBox(room_dim, fs=16000, max_order=15, absorption=0.35, sigma2_awgn=1e-8)

# add mic and good source to room
# Add silent signals to all sources
for sig, d, loc in zip(signals, delays, locations):
    room.add_source(loc, signal=sig, delay=d)

# add microphone array
room.add_microphone_array(pra.MicrophoneArray(np.c_[[6.5, 4.49], [6.5, 4.51]], room.fs))

# Simulate
# The premix contains the signals before mixing at the microphones
# shape=(n_sources, n_mics, n_samples)
separate_recordings = room.simulate(return_premix=True)

# Mix down the recorded signals (n_mics, n_samples)
# i.e., just sum the array over the sources axis
mics_signals = np.sum(separate_recordings, axis=0)

# STFT parameters
L = 2048
hop = L // 4
win_a = pra.hamming(L)
win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

# Observation vector in the STFT domain
X = pra.transform.stft.analysis(mics_signals.T, L, hop, win=win_a)

# Reference signal to calculate performance of BSS
ref = separate_recordings[:, 0, :]
SDR, SIR = [], []


Y = pra.bss.auxiva(X, n_iter=30, proj_back=True, callback=convergence_callback)

# run iSTFT
y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
y = y[L - hop:, :].T
m = np.minimum(y.shape[1], ref.shape[1])

# Compare SIR and SDR with our reference signal
sdr, sir, sar, perm = bss_eval_sources(ref[:,:m], y[:,:m])

wavfile.write('sep_example1.wav', 16000, y[perm[0],:].astype(np.float32))
wavfile.write('sep_example2.wav', 16000, y[perm[1],:].astype(np.float32))
fig = plt.figure()
plt.subplot(2,2,1)
plt.specgram(ref[0,:], NFFT=1024, Fs=room.fs)
plt.title('Source 0 (clean)')

plt.subplot(2,2,2)
plt.specgram(ref[1,:], NFFT=1024, Fs=room.fs)
plt.title('Source 1 (clean)')

plt.subplot(2,2,3)
plt.specgram(y[perm[0],:], NFFT=1024, Fs=room.fs)
plt.title('Source 0 (separated)')

plt.subplot(2,2,4)
plt.specgram(y[perm[1],:], NFFT=1024, Fs=room.fs)
plt.title('Source 1 (separated)')

plt.tight_layout(pad=0.5)

fig = plt.figure()
a = np.array(SDR)
b = np.array(SIR)
plt.plot(np.arange(a.shape[0]) * 10, a[:,0], label='SDR Source 0', c='r', marker='*')
plt.plot(np.arange(a.shape[0]) * 10, a[:,1], label='SDR Source 1', c='r', marker='o')
plt.plot(np.arange(b.shape[0]) * 10, b[:,0], label='SIR Source 0', c='b', marker='*')
plt.plot(np.arange(b.shape[0]) * 10, b[:,1], label='SIR Source 1', c='b', marker='o')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('dB')

plt.show()