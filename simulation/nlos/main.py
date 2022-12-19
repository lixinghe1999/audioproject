import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from typing import Tuple
import pyroomacoustics as pra
try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err


np.random.seed(42)
USE_RAYTRACING = False

SAMPLING_RATE = 44100
SPEED_OF_SOUND = 343  # m/s
sound_delay_samples_per_meter = 1 / SPEED_OF_SOUND * SAMPLING_RATE
SOUND_DELAY_SAMPLES_CONSTANT = 40.5  # determined empirically, matches (pra.constants.get("frac_delay_length") / 2)

SIGNAL_LENGTH = 100


def construct_room_from_file(filename: str, raytracing: bool) -> pra.Room:

    material_absorber = pra.Material(energy_absorption=1.0, scattering=0.0)
    material_mirror = pra.Material(energy_absorption=0.0, scattering=0.0)
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
        if material_type == 'mirror':
            material = material_mirror
        elif material_type == 'absorber':
            material = material_absorber
        else:
            raise ValueError(f"Unexpected wall material: {material_type}")
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
        ray_tracing=False,
        air_absorption=False,
    )
    if raytracing:
        room.set_ray_tracing(hist_bin_size=1 / SAMPLING_RATE)
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


def verify(signal: np.ndarray, source: np.ndarray, mic: np.ndarray, nlos_only: bool) -> Tuple[np.ndarray, np.ndarray]:

    room = construct_room_from_file("room_nlos" if nlos_only else "room_los_plus_nlos", raytracing=USE_RAYTRACING)
    # add source and mic and compute rir
    room.add_source(source, signal=signal)
    room.add_microphone(mic)
    room.compute_rir()
    room.simulate()
    # compute reflection point of single signal path
    reflection_point = compute_reflection_point(source, mic)
    # plot room and signal path
    plot_room_birdseye(source, mic, reflection_point, nlos_only=nlos_only)
    # compute path length in metres
    distance_nlos = np.linalg.norm(source - reflection_point) + np.linalg.norm(reflection_point - mic)
    # compute and apply time-of-flight
    offset_nlos = int(SOUND_DELAY_SAMPLES_CONSTANT + sound_delay_samples_per_meter * distance_nlos)

    if nlos_only:
        # apply attenuation
        truth = signal / distance_nlos
        # extract relevant part of prediction
        prediction = room.mic_array.signals[0, offset_nlos:offset_nlos + len(signal)]
    else:
        distance_los = np.linalg.norm(source - mic)
        offset_los = int(SOUND_DELAY_SAMPLES_CONSTANT + sound_delay_samples_per_meter * distance_los)
        offset_difference = offset_nlos - offset_los
        truth = (
                np.pad(signal, pad_width=[0, offset_difference]) / distance_los
                +
                np.pad(signal, pad_width=[offset_difference, 0]) / distance_nlos
        )
        # extract relevant part of prediction
        prediction = room.mic_array.signals[0, offset_los:offset_nlos + len(signal)]
    return truth, prediction


def exec_one_test(signal, mic, nlos_only: bool, source: np.ndarray = np.array([1, 1, 1])):

    truth, prediction = verify(signal, source=source, mic=mic, nlos_only=nlos_only)

    ax = plt.gca()
    ax.plot(truth, label="truth")
    ax.plot(prediction, label="pred")
    ax.legend()
    plt.title(f"source: [1, 1, 1], mic: {list(mic)}")
    plt.show()


def main():
    # simple sine wave as test signal
    signal = np.sin(np.asarray(range(SIGNAL_LENGTH)) * np.pi / 10)

    """
    basic issue: wrong signal amplitude ofr NLOS-path when Y and Z coordinates od source and mic match
    """
    # nlos only, Y and Z of mic match source -> wrong amplitude
    exec_one_test(signal, mic=np.array([3, 1, 1]), nlos_only=True)
    # nlos only, Y differs by 1mm -> correct amplitude
    exec_one_test(signal, mic=np.array([3, 1.001, 1]), nlos_only=True)
    # nlos only, Z differs by 1mm -> correct amplitude
    # exec_one_test(signal, mic=np.array([3, 1, 1.001]), nlos_only=True)
    """
    also in LOS+NLOS, wrong amplitude for nlos part of signal
    """
    # nlos plus los, Y and Z of mic match source -> wrong amplitude
    exec_one_test(signal, mic=np.array([3, 1, 1]), nlos_only=False)
    # nlos plus los, Y differs by 1mm -> nearly correct amplitude (perfect when not using raytracing)
    exec_one_test(signal, mic=np.array([3, 1.001, 1]), nlos_only=False)
    # nlos plus los, Z differs by 1mm -> nearly correct amplitude (perfect when not using raytracing)
    # exec_one_test(signal, mic=np.array([3, 1, 1.001]), nlos_only=False)
    """
    weird: at Y=1.5 (but not at Y=1.0) the issue also surfaces if Y and Z both differ by 1mm
    """
    # nlos plus los, Y and Z differ by 1mm -> nearly correct amplitude (perfect when not using raytracing)
    exec_one_test(signal, mic=np.array([3, 1.001, 1.001]), nlos_only=False)
    # nlos plus los, Y and Z differ by 1mm, different Y position -> wrong amplitude again
    exec_one_test(signal, mic=np.array([3, 1.501, 1.001]), nlos_only=False, source=np.array([1, 1.5, 1]))
    # nlos plus los, only Y differs by 1mm, different Y position -> nearly correct amplitude (perfect when not using raytracing)
    exec_one_test(signal, mic=np.array([3, 1.501, 1]), nlos_only=False, source=np.array([1, 1.5, 1]))


if __name__ == "__main__":
    main()
