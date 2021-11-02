import os
import random
from micplot import get_wav, save_wav
import matplotlib.pyplot as plt


if __name__ == "__main__":
    noise_path = 'reference/conversation.wav'
    fig, axs = plt.subplots(3, 1)
    wave_noise, Zxx, _ = get_wav(noise_path, normalize=False)
    path = 'exp3/mic1/'
    save_path = 'exp3/noisy1/'
    files = os.listdir(path)
    length = len(wave_noise)
    for file in files:
        wave_clean, Zxx_clean, _ = get_wav(path + file, normalize=False)
        random_index = random.randint(0, length - len(wave_clean))
        ratio = wave_clean.max()/wave_noise.max()
        # keep the same volume
        save_wav(wave_clean + ratio * wave_noise[random_index: random_index+len(wave_clean)], save_path + file)

