import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
def init_room(room_dim=[5, 5, 2]):
    # The desired reverberation time and dimensions of the room
    rt60 = 0.5  # seconds
    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order,
    )
    mic_loc = np.c_[
        [m/2 for m in room_dim]
    ]
    # finally place the array in the room
    room.add_microphone_array(mic_loc)
    return room
if __name__ == "__main__":

    num_rir = 200
    room_dim = [[5, 5, 2], [10, 10, 2], [20, 20, 2]]
    sizes = ['small/', 'middle/', 'large/']
    for dim, size in zip(room_dim, sizes):
        room = init_room(dim)
        random_loc = np.append(np.random.random(2) * dim[0], np.random.random() * 2)
        delay = np.random.random() * 1
        room.add_source(random_loc)
        room.compute_rir()
        wav.write(size + 'rir.wav', 16000, room.rir[0][0])
        for i in range(num_rir):
            room = init_room(dim)
            random_loc = np.append(np.random.random(2) * dim[0], np.random.random() * 2)
            delay = np.random.random() * 1
            room.add_source(random_loc, delay=delay)
            room.compute_rir()
            wav.write(size  + str(i) + '_rir.wav', 16000, room.rir[0][0])
            # plt.plot(room.rir[0][0])
            # plt.show()