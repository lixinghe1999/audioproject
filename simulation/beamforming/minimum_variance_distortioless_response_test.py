# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 17:49:51 2019

@author: a-kojima
"""
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from beamformer import util
from beamformer import minimum_variance_distortioless_response as mvdr

SAMPLING_FREQUENCY = 16000
FFT_LENGTH = 512
FFT_SHIFT = 256
ENHANCED_WAV_NAME = './output/enhanced_speech_mvdr.wav'
MIC_ANGLE_VECTOR = np.array([0, 60, 120, 180, 240, 300])
LOOK_DIRECTION = 0
MIC_DIAMETER = 0.1

def multi_channel_read(prefix=r'./sample_data/20G_20GO010I_STR.CH{}.wav',
                       channel_index_vector=np.array([1, 2, 3, 4, 5, 6])):
    wav, _ = sf.read(prefix.replace('{}', str(channel_index_vector[0])), dtype='float32')
    wav_multi = np.zeros((len(wav), len(channel_index_vector)), dtype=np.float32)
    wav_multi[:, 0] = wav
    for i in range(1, len(channel_index_vector)):
        wav_multi[:, i] = sf.read(prefix.replace('{}', str(channel_index_vector[i])), dtype='float32')[0]
    return wav_multi

def MVDR(LOOK_DIRECTION):
    # multi_channels_data = multi_channel_read()
    multi_channels_data, _ = sf.read('noisy.wav')

    complex_spectrum, _ = util.get_3dim_spectrum_from_data(multi_channels_data, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
    mvdr_beamformer = mvdr.minimum_variance_distortioless_response(MIC_ANGLE_VECTOR, MIC_DIAMETER,
                                                                   sampling_frequency=SAMPLING_FREQUENCY,
                                                                   fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)
    steering_vector = mvdr_beamformer.get_sterring_vector(LOOK_DIRECTION)
    spatial_correlation_matrix = mvdr_beamformer.get_spatial_correlation_matrix(multi_channels_data)
    beamformer = mvdr_beamformer.get_mvdr_beamformer(steering_vector, spatial_correlation_matrix)
    enhanced_speech = mvdr_beamformer.apply_beamformer(beamformer, complex_spectrum)
    enhanced_speech = enhanced_speech / np.max(enhanced_speech)
    return enhanced_speech

reference_speech = sf.read('example.wav')[0]
reference_speech = reference_speech / np.max(reference_speech)

for angle in [0, 90, 180, 270]:
    enhanced_speech = MVDR(LOOK_DIRECTION=angle)[:80000]
    error = np.mean(np.abs(reference_speech - enhanced_speech))
    print(angle, error)
    fig, axs = plt.subplots(2)
    axs[0].plot(reference_speech)
    axs[1].plot(enhanced_speech)
    plt.show()
#sf.write(ENHANCED_WAV_NAME, enhanced_speech, SAMPLING_FREQUENCY)

