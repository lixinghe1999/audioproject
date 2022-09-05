import os
import librosa
from Voice_Activity_Detector.vad import VAD
from resemblyzer import preprocess_wav, VoiceEncoder
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

def segment_embedding(token, data1, data2, data3, embeddings):
    # vad = VAD(data3, rate_mic, nFFT=512, win_length=0.025, hop_length=0.01, theshold=0.999)
    for j in range(int((T - segment) / segment) + 1):
        start = j * segment
        stop = (j + 1) * segment
        imu1 = data1[rate_imu * start: rate_imu * stop]
        imu2 = data2[rate_imu * start: rate_imu * stop]
        audio = data3[rate_mic * start: rate_mic * stop]

        embedding = update_phones(token, imu1, imu2, audio, start, stop)

        # corr1 = matching_features(data3, data1)
        # corr2 = matching_features(data3, data2)
        # embedding = np.concatenate([corr1, corr2], axis=1)

        # embedding = get_spk_emb(data3)

        # bcf1 = estimate_response(data3, data1)
        # bcf2 = estimate_response(data3, data2)
        # embedding = np.concatenate([bcf1[0, :] / np.max(bcf1[0, :]), bcf2[0, :] / np.max(bcf2[0, :])])

        # embedding = np.mean(librosa.feature.mfcc(y=data3, sr=rate_mic), axis=1)
        embeddings.append(embedding)
    return embeddings

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=0, required=False)
    args = parser.parse_args()


    directory = '../dataset/our'
    person = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
    dict = {}
    for k in range(len(person)):
        print(k)
        p = person[k]
        path = os.path.join(directory, p, 'train')
        file_list = os.listdir(path)
        dict = directory_decompose(file_list)
        imu1 = dict['bmiacc1']
        imu2 = dict['bmiacc2']
        audio1 = dict['mic']
        if '新录音' in dict:
            audio2 = dict['新录音']
        embeddings = []
        dataset = {}
        model = read_recognizer()
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

            token = model.recognize(os.path.join(path, audio1[i]), 'eng', timestamp=True)
            token = token.split('\n')
            segment_embedding(token, data1, data2, data3, embeddings)
        json.dump(embeddings, open('speaker_embedding/phone_embedding/' + str(k) + '.json', 'w'), indent=4)

        # embeddings = np.stack(embeddings, axis=0)
        # np.save('speaker_embedding/DNN_embedding/' + str(k), embeddings)
        # np.save('speaker_embedding/bcf_embedding/' + str(k), embeddings)
        # np.save('speaker_embedding/mfcc_embedding/' + str(k), embeddings)
        # np.save('speaker_embedding/corr_embedding/' + str(k), embeddings)



