from scipy.io import wavfile
import tables
import numpy as np

#read data from wav
fs, data = wavfile.read('./test.wav')

#ouput
folder='./'
name= '1.h5'
#save_to acoular h5 format
acoularh5 = tables.open_file(folder+name, mode = "w", title = name)
data = np.expand_dims(data, axis=1)
print(data.shape)
acoularh5.create_earray('/', 'time_data', atom=None, title='', filters=None, \
                         expectedrows=100000, byteorder=None, createparents=False, obj=data)
acoularh5.set_node_attr('/time_data','sample_freq', fs)
acoularh5.close()