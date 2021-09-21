import wave
import numpy as np
import matplotlib.pyplot as plt
def get_wav(name):
    f = wave.open(name, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print(nchannels, sampwidth, framerate, nframes)
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.frombuffer(str_data, dtype = np.short)
    time = np.arange(0, nframes)/framerate
    return wave_data, time
plt.figure(1)
plt.subplot(2,1,1)
wave_1, time = get_wav('mic1.wav')
plt.plot(time, wave_1)
plt.subplot(2,1,2)
wave_2, time = get_wav('mic2.wav')
plt.plot(time, wave_2)
plt.xlabel("time")
plt.show()
