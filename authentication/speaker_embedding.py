import os
import librosa
from resemblyzer import preprocess_wav, VoiceEncoder, normalize_volume, trim_long_silences
from bone_conduction_function import estimate_response, matching_features, update_phones
import numpy as np
import torch
import scipy.signal as signal
import argparse
from voicefilter import SpeechEmbedder
from vggvox.vggm import VGGM
from allosaurus.app import read_recognizer

T = 5
segment = 5
rate_imu = 1600
rate_mic = 16000
def init_model(model_name = 'resemblyzer'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'resemblyzer':
        model = VoiceEncoder(device=device, verbose=False)
    elif model_name == 'voicefilter':
        model = SpeechEmbedder().to('cuda')
        ckpt = torch.load('embedder.pt')
        model.load_state_dict(ckpt)
    else:# VGGVOX
        model = VGGM().to('cuda')
        ckpt = torch.load('VGGM300_BEST_140_81.99.pth')
        model.load_state_dict(ckpt)
    return model
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

def segment_level_feature(model, data1, data2, data3, embeddings):
    for j in range(int((T - segment) / segment) + 1):
        start = j * segment
        stop = (j + 1) * segment
        imu1 = data1[rate_imu * start: rate_imu * stop]
        imu2 = data2[rate_imu * start: rate_imu * stop]
        audio = data3[rate_mic * start: rate_mic * stop]

        # Resemblyzer, VGGVOX, VoiceFilter
        # embedding = model.embed_utterance(audio)
        with torch.no_grad():
            embedding = model(np.expand_dims(audio, 0)).cpu().numpy()[0]
        # bcf1 = estimate_response(audio, imu1)
        # bcf2 = estimate_response(audio, imu2)
        # embedding = np.stack([bcf1, bcf2], axis=0)
        embeddings.append(embedding)
    return embeddings

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=0, required=False)
    args = parser.parse_args()

    names = ['resemblyzer', 'voicefilter', 'vggvox']
    model_name = names[1]
    model = init_model(model_name)
    store_path = 'speaker_embedding/' + model_name
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    if args.mode == 0:
        # audio-imu dataset
        directory = '../dataset/our'
        person = ["1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he", "hou"]
        for k in range(len(person)):
            count = 0
            print(k)
            p = person[k]
            embeddings = []
            for s in ['s1']:
                path = os.path.join(directory, p, s, 'noise')
                file_list = os.listdir(path)
                dict = directory_decompose(file_list)
                imu1 = dict['bmiacc1']
                imu2 = dict['bmiacc2']
                audio1 = dict['mic']
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
                    data3 = signal.filtfilt(b, a, data3, axis=0)
                    data3 = normalize_volume(data3, -30, increase_only=True)
                    embeddings = segment_level_feature(model, data1, data2, data3, embeddings)
            embeddings = np.array(embeddings)
            print(embeddings.shape)
            np.save(store_path + '/' + str(k), embeddings)
    elif args.mode == 1:
        directory = '../dataset/librispeech-100'
        person = os.listdir(directory)
        for p in person:
            count = 0
            print(p)
            embeddings = []
            path_p = os.path.join(directory, p)
            for s in os.listdir(path_p):
                path = os.path.join(path_p, s)
                file_list = os.listdir(path)
                file_list.sort()
                max_sample = min(10, len(file_list)-1)
                for i in range(len(file_list[:max_sample])):
                    b, a = signal.butter(4, 80, 'highpass', fs=rate_mic)
                    audio, source_sr = librosa.load(os.path.join(path, file_list[i]), sr=None)
                    audio = signal.filtfilt(b, a, audio, axis=0)
                    audio = normalize_volume(audio, -30, increase_only=True)
                    audio = trim_long_silences(audio)
                    if len(audio) < 80000:
                        continue
                    else:
                        # Resemblyzer, VGGVOX, VoiceFilter
                        # embedding = model.embed_utterance(audio)
                        with torch.no_grad():
                            embedding = model(np.expand_dims(audio, 0)).cpu().numpy()[0]
                        embeddings.append(embedding)
            embeddings = np.array(embeddings)
            print(embeddings.shape)
            np.save(store_path + '/' + p, embeddings)


