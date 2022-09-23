import os
import librosa
from Voice_Activity_Detector.vad import VAD
from resemblyzer import preprocess_wav, VoiceEncoder, normalize_volume
from bone_conduction_function import estimate_response, matching_features, update_phones
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal as signal
import argparse
from allosaurus.app import read_recognizer
import json


T = 30
segment = 3
rate_imu = 1600
rate_mic = 16000

def get_spk_emb(wav):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resemblyzer_encoder = VoiceEncoder(device=device, verbose=False)
    embeds = resemblyzer_encoder.embed_utterance(wav)
    return embeds

def directory_decompose(files):
    import itertools
    dict = {}
    for k, v in itertools.groupby(files, key=lambda x: x.replace(' ', '_').split('_')[0]):
        dict[k] = list(v)
    return dict

def phone_level_feature(model, data1, data2, data3, embeddings):
    token = model.recognize(os.path.join(path, audio1[i]), 'eng', timestamp=True)
    token = token.split('\n')
    for j in range(int((T - segment) / segment) + 1):
        start = j * segment
        stop = (j + 1) * segment
        imu1 = data1[rate_imu * start: rate_imu * stop]
        imu2 = data2[rate_imu * start: rate_imu * stop]
        audio = data3[rate_mic * start: rate_mic * stop]
        embeddings = update_phones(token, imu1, imu2, audio, start, stop, embeddings)
    return embeddings

def segment_level_feature(data1, data2, data3, count, store_path):
    for j in range(int((T - segment) / segment) + 1):
        start = j * segment
        stop = (j + 1) * segment
        imu1 = data1[rate_imu * start: rate_imu * stop]
        imu2 = data2[rate_imu * start: rate_imu * stop]
        audio = data3[rate_mic * start: rate_mic * stop]

        # corr1 = matching_features(data3, data1)
        # corr2 = matching_features(data3, data2)
        # embedding = np.concatenate([corr1, corr2], axis=1)

        # embedding = get_spk_emb(data3)

        bcf1 = estimate_response(audio, imu1)
        bcf2 = estimate_response(audio, imu2)
        embedding = np.stack([bcf1[:, :], bcf2[:, :]], axis=0)
        # embedding = np.mean(librosa.feature.mfcc(y=data3, sr=rate_mic), axis=1)
        np.save(store_path + '/' + str(count), embedding)
        count += 1
    return count

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=0, required=False)
    args = parser.parse_args()


    directory = '../dataset/our'
    person = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
    dict = {}
    for k in range(len(person)):
        count = 0
        print(k)
        p = person[k]
        path = os.path.join(directory, p, 'noise_train')
        file_list = os.listdir(path)
        print(file_list)
        dict = directory_decompose(file_list)
        imu1 = dict['bmiacc1']
        imu2 = dict['bmiacc2']
        audio1 = dict['mic']
        if '新录音' in dict:
            audio2 = dict['新录音']
        embeddings = []
        dataset = {}

        store_path = 'speaker_embedding/noise_DNN_embedding/' + str(k)
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        if args.mode == 0:
            model = read_recognizer()
            embeddings = {}
            for i in range(len(imu1)):
                b, a = signal.butter(4, 80, 'highpass', fs=rate_imu)
                data1 = np.loadtxt(os.path.join(path, imu1[i])) / 2 ** 14
                data2 = np.loadtxt(os.path.join(path, imu2[i])) / 2 ** 14
                data1 = signal.filtfilt(b, a, data1, axis=0)
                data2 = signal.filtfilt(b, a, data2, axis=0)
                data1 = np.clip(data1, -0.05, 0.05)
                data2 = np.clip(data2, -0.05, 0.05)

                b, a = signal.butter(4, 80, 'highpass', fs=rate_mic)
                data3 = preprocess_wav(os.path.join(path, audio1[i]))
                data3 = signal.filtfilt(b, a, data3, axis=0)

                embeddings = phone_level_feature(model, data1, data2, data3, embeddings)
            json.dump(embeddings, open(store_path + '.json', 'w'), indent=4)
        else:
            print(imu1)
            for i in range(len(imu1)):
                b, a = signal.butter(4, 80, 'highpass', fs=rate_imu)
                data1 = np.loadtxt(os.path.join(path, imu1[i])) / 2 ** 14
                data2 = np.loadtxt(os.path.join(path, imu2[i])) / 2 ** 14
                data1 = signal.filtfilt(b, a, data1, axis=0)
                data2 = signal.filtfilt(b, a, data2, axis=0)
                data1 = np.clip(data1, -0.05, 0.05)
                data2 = np.clip(data2, -0.05, 0.05)

                b, a = signal.butter(4, 80, 'highpass', fs=rate_mic)

                data3, source_sr = librosa.load(os.path.join(path, audio1[i]), sr=None)
                data3 = normalize_volume(data3, -30, increase_only=True)
                data3 = signal.filtfilt(b, a, data3, axis=0)
                count = segment_level_feature(data1, data2, data3, count, store_path)


