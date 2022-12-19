import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from typing import Tuple
import pyroomacoustics as pra
from scipy.io import wavfile
from stl import mesh
from doa import DOA
'''
The numpy-stl package is required for this example. Install it with `pip install numpy-stl
'''

np.random.seed(42)
SAMPLING_RATE = 16000
SPEED_OF_SOUND = 343  # m/s
sound_delay_samples_per_meter = 1 / SPEED_OF_SOUND * SAMPLING_RATE
SOUND_DELAY_SAMPLES_CONSTANT = 40.5  # determined empirically, matches (pra.constants.get("frac_delay_length") / 2)
SIGNAL_LENGTH = 20000


def construct_room_from_file(filename: str) -> pra.Room:

    material_absorber = pra.Material(energy_absorption=0.2, scattering=0.2)
    material_mirror = pra.Material(energy_absorption=0.2, scattering=0.2)
    # load wall materials from file
    with open(f"{filename}.material", 'r') as f:
        m = [line.strip().split(';')[2] for line in f.readlines()]
    # with numpy-stl
    the_mesh = mesh.Mesh.from_file(f"{filename}.stl")
    ntriang, nvec, npts = the_mesh.vectors.shape
    # create one wall per triangle
    walls = []
    for w in range(ntriang):
        material_type = m[w]
        material = material_absorber
        # if material_type == 'mirror':
        #     material = material_mirror
        # elif material_type == 'absorber':
        #     material = material_absorber
        # else:
        #     raise ValueError(f"Unexpected wall material: {material_type}")
        walls.append(
            pra.wall_factory(
                the_mesh.vectors[w].T,
                material.energy_absorption["coeffs"],
                material.scattering["coeffs"],
            )
        )
    room = pra.Room(
        walls,
        fs=SAMPLING_RATE,
        max_order=1,
        air_absorption=False
    )
    #room.set_ray_tracing()
    #room.simulator_state['ism_needed'] = False
    print(room.simulator_state)
    return room


def plot_room_birdseye(source: np.ndarray, mic: np.ndarray, reflection_point: np.ndarray, nlos_only: bool):
    obstacle_offset = (1.5, 0.3)  # fixed in stl
    ax = plt.gca()
    rect = plt.Rectangle((0, 0), 4, 2, fc='white', ec='black')
    ax.add_patch(rect)
    if nlos_only:
        rect = plt.Rectangle(obstacle_offset, 1, 1, fc='white', ec='black')
        ax.add_patch(rect)
    line = plt.Line2D((0, 4), (2, 2), lw=1.5, c='green')
    ax.add_line(line)
    line = plt.Line2D((source[0], reflection_point[0]), (source[1], reflection_point[1]), lw=1.5)
    ax.add_line(line)
    line = plt.Line2D((mic[0], reflection_point[0]), (mic[1], reflection_point[1]), lw=1.5)
    ax.add_line(line)
    line = plt.Line2D((mic[0], source[0]), (mic[1], source[1]), lw=1.5, c=('red' if nlos_only else 'blue'))
    ax.add_line(line)
    ax.scatter(mic[0], mic[1], c='blue')
    ax.annotate('mic', (mic[0], mic[1]))
    ax.scatter(source[0], source[1], c='blue')
    ax.annotate('source', (source[0], source[1]))
    plt.axis('scaled')
    plt.show()


def compute_reflection_point(source: np.ndarray, mic: np.ndarray) -> np.ndarray:
    dist_x = mic[0] - source[0]
    source_to_wall_to_mic_to_wall_plus_source_to_wall = (2 - source[1]) / ((2 - mic[1]) + (2 - source[1]))
    source_to_reflection_point_x = dist_x * source_to_wall_to_mic_to_wall_plus_source_to_wall
    reflection_point = np.array([source[0] + source_to_reflection_point_x, 2, 1])
    return reflection_point


def simulate(signal: np.ndarray, nlos_only: bool) -> Tuple[np.ndarray, np.ndarray]:

    source = np.array([1, 1, 1])
    room = construct_room_from_file("room_nlos" if nlos_only else "room_los_plus_nlos")
    # add source and mic and compute rir
    room.add_source(position=source, signal=signal)
    center = [2, 1, 1]
    radius = 0.1
    N = 8 / 2
    mic = np.c_[
        [center[0] + radius * np.cos(0 * np.pi / N), center[1] + radius * np.sin(0 * np.pi / N), center[2]],  # Mic 1
        [center[0] + radius * np.cos(1 * np.pi / N), center[1] + radius * np.sin(1 * np.pi / N), center[2]],  # Mic 2
        [center[0] + radius * np.cos(2 * np.pi / N), center[1] + radius * np.sin(2 * np.pi / N), center[2]],  # Mic 3
        [center[0] + radius * np.cos(3 * np.pi / N), center[1] + radius * np.sin(3 * np.pi / N), center[2]],  # Mic 4
        [center[0] + radius * np.cos(4 * np.pi / N), center[1] + radius * np.sin(4 * np.pi / N), center[2]],  # Mic 5
        [center[0] + radius * np.cos(5 * np.pi / N), center[1] + radius * np.sin(5 * np.pi / N), center[2]],  # Mic 6
        [center[0] + radius * np.cos(6 * np.pi / N), center[1] + radius * np.sin(4 * np.pi / N), center[2]],  # Mic 7
        [center[0] + radius * np.cos(7 * np.pi / N), center[1] + radius * np.sin(5 * np.pi / N), center[2]],  # Mic 8
    ]
    room.add_microphone(mic)
    room.simulate()
    # room.plot_rir()
    # plt.show()
    mic_1 = mic[:, 0]
    # compute reflection point of single signal path
    reflection_point = compute_reflection_point(source, mic_1)
    # plot room and signal path
    # plot_room_birdseye(source, mic, reflection_point, nlos_only=nlos_only)
    # compute path length in metres
    distance_nlos = np.linalg.norm(source - reflection_point) + np.linalg.norm(reflection_point - mic_1)
    # compute and apply time-of-flight
    offset_nlos = int(SOUND_DELAY_SAMPLES_CONSTANT + sound_delay_samples_per_meter * distance_nlos)

    if nlos_only:
        # apply attenuation
        truth = signal / distance_nlos
        # extract relevant part of prediction
        prediction = room.mic_array.signals[:, offset_nlos:offset_nlos + len(signal)]
    else:
        distance_los = np.linalg.norm(source - mic_1)
        offset_los = int(SOUND_DELAY_SAMPLES_CONSTANT + sound_delay_samples_per_meter * distance_los)
        offset_difference = offset_nlos - offset_los
        truth = (
                np.pad(signal, pad_width=[0, offset_difference]) / distance_los
                +
                np.pad(signal, pad_width=[offset_difference, 0]) / distance_nlos
        )
        # extract relevant part of prediction
        prediction = room.mic_array.signals[:, offset_los:offset_nlos + len(signal)]
    # fig, axs = plt.subplots(2)
    # axs[0].plot(truth, c='red')
    # axs[1].plot(prediction.T, c='blue')
    # plt.show()
    azimuth = np.array([180.]) / 180. * np.pi
    DOA(room.mic_array.signals.T, mic, azimuth, 1)
    return prediction.T

if __name__ == "__main__":
    # simple sine wave as test signal
    #fs, signal = wavfile.read("example.wav")
    t = np.linspace(0, SIGNAL_LENGTH, SIGNAL_LENGTH)
    signal = np.sin(t)
    #mic = pra.MicrophoneArray(R, fs=SAMPLING_RATE)
    signals = simulate(signal, nlos_only=False)
    print(signals.shape)
    for i in range(8):
        wavfile.write('data/nlos_los_ch{}.wav'.format(str(i+1)), SAMPLING_RATE, signals[:, i].astype(np.int16))
    #exec_one_test(signal, mic=mic, nlos_only=True)
