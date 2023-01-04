import numpy as np
import matplotlib.pyplot as plt
def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    return x
data_4 = np.load('4batch.npy')
data_16 = np.load('16batch.npy')
data_4 = softmax(data_4)
data_16 = softmax(data_16)
fig, axs = plt.subplots(2)
axs[0].plot(data_4[:8, :].T)
axs[1].plot(data_16[:8, :].T)
plt.show()