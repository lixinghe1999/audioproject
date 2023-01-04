import numpy as np

data_4 = np.load('4batch.npy', allow_pickle=True)
data_16 = np.load('16batch.npy', allow_pickle=True)
print(data_4.shape, data_16.shape)