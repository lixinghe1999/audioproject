import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
def init_room():
    # The desired reverberation time and dimensions of the room
    rt60 = 0.5  # seconds
    room_dim = [5, 5, 1]  # meters

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order,
    )
    mic_loc = np.c_[
        [2.5, 2.5, 0.5]
    ]
    # finally place the array in the room
    room.add_microphone_array(mic_loc)
    return room
if __name__ == "__main__":

    num_rir = 200
    for i in range(100, num_rir):
        room = init_room()
        random_loc = np.append(np.random.random(2) * 5, np.random.random())
        delay = np.random.random() * 2
        room.add_source(random_loc, delay=delay)
        room.compute_rir()
        wav.write(str(i) + '_rir.wav', 16000, room.rir[0][0])
        # plt.plot(room.rir[0][0])
        # plt.show()