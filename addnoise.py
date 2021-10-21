import os
import random
from micplot import get_wav, save_wav
import matplotlib.pyplot as plt


if __name__ == "__main__":
    noise_path = 'reference/conversation.wav'
    fig, axs = plt.subplots(3, 1)
    wave_noise, Zxx, _ = get_wav(noise_path, normalize=False)
    path = 'exp2/HOU/mic2/'
    save_path = 'exp2/HOU/noisy2/'
    files = os.listdir(path)
    for file in ['noise_test.wav']:
        wave_clean, Zxx_clean, _ = get_wav(file, normalize=False)
        random_index = random.randint(0, 1584000)
        save_wav(wave_clean + 1.6 * wave_noise[random_index: random_index+len(wave_clean)], 'noise_test.wav')

