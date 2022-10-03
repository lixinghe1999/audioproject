import os
import matplotlib.cm as cm
from matplotlib.colors import Normalize


import matplotlib.pyplot as plt
import numpy as np

cmap = cm.gray
norm = Normalize(vmin=0, vmax=14)
path = 'speaker_embedding/noise_dependent'
fig, axs = plt.subplots(2,1)
for i, p in enumerate(os.listdir(path)):
    print(p)
    person_path = os.path.join(path, p)
    files = os.listdir(person_path)
    for f in files:
        data = np.load(os.path.join(person_path, f))
        axs[0].plot(data[0]/np.max(data[0]), c=cmap(norm(i)))
        axs[1].plot(data[1]/np.max(data[1]), c=cmap(norm(i)))
plt.show()