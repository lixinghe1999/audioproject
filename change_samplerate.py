import os
import random
from micplot import get_wav, save_wav
import matplotlib.pyplot as plt


if __name__ == "__main__":
    noise_path = 'reference/conversation.wav'
    fig, axs = plt.subplots(3, 1)
    wave_noise, Zxx, _ = get_wav(noise_path, normalize=False)
    path = 'exp2/HE/mic1'
    files = os.listdir(path)
    for file in files:
        if file[-4:] == '.wav':
            wave_clean, Zxx_clean, _ = get_wav(path + file, normalize=False)
            random_index = random.randint(0, 4300000)
            wave_noise = wave_clean + 0.1 * wave_noise[random_index: random_index+len(wave_clean)]
            save_wav(wave_noise, 'exp2/HE/noisy1/' + file)

