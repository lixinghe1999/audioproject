import numpy as np
tfs = np.load('transfer_function_EMSB_32.npy')
print(tfs.shape)
index = []
for i, tf in enumerate(tfs):
    f = tf[0]
    v = tf[1]
    if np.mean(f) < 2 * np.mean(v) or np.max(f) == 0:
        print('found unstable')
    else:
        index.append(i)
tfs_new = tfs[index]
np.save('transfer_function_EMSB_filter.npy', tfs_new, )
print(tfs_new.shape)
