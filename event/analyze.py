import numpy as np
import matplotlib.pyplot as plt
data_4 = np.load('4batch.npy')
data_16 = np.load('16batch.npy')
fig, axs = plt.subplots(2)
axs[0].plot(data_4[:4, :].T)
axs[1].plot(data_16[:4, :].T)
plt.show()