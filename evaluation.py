from pesq import pesq
import micplot
import os
import numpy as np
import matplotlib.pyplot as plt
reference = 'exp2/HE/'
noisy = 'exp2/noisy/'
index = 1
rate = 16000
def ab_gap(ref, deg):
    #return pesq(rate, ref, deg, 'wb')
    return np.mean(np.abs(ref-deg[:len(ref)]))

ref, _, _ = micplot.get_wav(reference + os.listdir(reference)[index+27], normalize=True)
noise, _, _ = micplot.get_wav(noisy + os.listdir(noisy)[index], normalize=True)
re1, _, _ = micplot.get_wav("test.wav", normalize=True)
re2, _, _ = micplot.get_wav("reduced_test.wav", normalize=True)
plt.plot(ref)
#plt.plot(noise)
#plt.plot(re1)
plt.plot(re2)
plt.show()
print(ab_gap(ref, noise))
print(ab_gap(ref, re1))
print(ab_gap(ref, re2))
