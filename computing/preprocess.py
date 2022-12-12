import numpy as np
from scipy.signal import lfilter, stft

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    print(sig.shape, refsig.shape)
    n = sig.shape[0] + refsig.shape[0]
    print(n)
    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    print(SIG.shape, REFSIG.shape)
    R = SIG * np.conj(REFSIG)
    print(R.shape)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    print(cc.shape)
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    print(max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    print(cc.shape)
    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau, cc


def rm_dc_n_dither(audio):
    # All files 16kHz tested..... Will copy for 8kHz from author's matlab code later
    alpha = 0.99
    b = [1, -1]
    a = [1, -alpha]

    audio = lfilter(b, a, audio)

    dither = np.random.uniform(low=-1, high=1, size=audio.shape)
    spow = np.std(audio)
    return audio + (1e-6 * spow) * dither


def preemphasis(audio, alpha=0.97):
    b = [1, -alpha]
    a = 1
    return lfilter(b, a, audio)


def normalize_frames(m, epsilon=1e-12):
    return (m - m.mean(1, keepdims=True)) / np.clip(m.std(1, keepdims=True), epsilon, None)


def preprocess(audio, buckets=None, sr=16000, Ws=25, Ss=10, alpha=0.97, cross=False):
    # ms to number of frames
    if not buckets:
        buckets = {100: 2,
                   200: 5,
                   300: 8,
                   400: 11,
                   500: 14,
                   600: 17,
                   700: 20,
                   800: 23,
                   900: 27,
                   1000: 30}

    Nw = round((Ws * sr) / 1000)
    Ns = round((Ss * sr) / 1000)
    # hamming window func signature
    window = np.hamming
    # get next power of 2 greater than or equal to current Nw
    nfft = 1 << (Nw - 1).bit_length()

    # Remove DC and add small dither
    audio = rm_dc_n_dither(audio)

    # Preemphasis filtering
    audio = preemphasis(audio, alpha)

    # get 512x300 spectrograms
    _, _, mag = stft(audio,
                     fs=sr,
                     window=window(Nw),
                     nperseg=Nw,
                     noverlap=Nw - Ns,
                     nfft=nfft,
                     return_onesided=False,
                     padded=False,
                     boundary=None)

    mag = normalize_frames(np.abs(mag))
    # Get the largest bucket smaller than number of column vectors i.e. frames
    rsize = max(i for i in buckets if i <= mag.shape[1])
    rstart = (mag.shape[1] - rsize) // 2
    # Return truncated spectrograms

    return mag[:, rstart:rstart + rsize]




if __name__ == "__main__":
    delay = 0.5
    fs = 16000

    refsig = np.linspace(0, 1, 16000)
    sig = np.concatenate((np.linspace(0, 0, int(delay * fs)), refsig))
    refsig = np.concatenate((refsig, np.linspace(0, 0, int(delay * fs))))


    offset, cc = gcc_phat(sig, refsig, fs=16000, interp=1)
    print(offset)
    plt.plot(sig, c='b')
    plt.plot(refsig, c='r')
    #plt.plot(cc, c='g')
    plt.show()



