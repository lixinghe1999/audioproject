import micplot
import noisereduce as nr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig, axs = plt.subplots(2,1)
    data, _, _ = micplot.get_wav('test.wav')
    rate = 44100
    reduced_noise = nr.reduce_noise(y= data, sr=rate, thresh_n_mult_nonstationary=1, stationary=False)
    #reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0, stationary=False)
    micplot.save_wav(reduced_noise, 'reduced_test.wav')
    axs[0].plot(data)
    axs[1].plot(reduced_noise)
    plt.show()