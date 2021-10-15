import os
import random
from micplot import get_wav
import matplotlib.pyplot as plt
import wave

if __name__ == "__main__":
    noise_path = 'reference/conversation.wav'
    fig, axs = plt.subplots(3, 1)
    wave_noise, Zxx = get_wav(noise_path)
    wave1_noise = wave_noise[::2]
    wave2_noise = wave_noise[1::2]
    path = 'exp2/HE/'
    files = os.listdir(path)
    for file in files:
        if file[-4:] == '.wav':
            wave_clean, Zxx_clean = get_wav(path + file)
            random_index = random.randint(0, 4300000)
            wave_noise = wave_clean + 0.2 * wave1_noise[random_index: random_index+len(wave_clean)]
            wf = wave.open('exp2/noisy/' + 'noisy' + file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(wave_noise.astype('int16'))
            wf.close()

