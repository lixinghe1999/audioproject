import argparse
import json
import os
import librosa
import torchaudio
import numpy as np
import soundfile as sf

def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):

    file_infos = []
    in_dir = os.path.abspath(in_dir)  # 返回绝对路径
    wav_list = os.listdir(in_dir)  # 返回该目录下文件清单

    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):  # 判断是否以 .wav 结尾
            continue
        wav_path = os.path.join(in_dir, wav_file)  # 拼接路径
        file_infos.append((wav_path, torchaudio.info(wav_path).num_frames))
    if not os.path.exists(out_dir):  # 如果输出路径不存在，就创造该路径
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)  # 将信息写入 json


def preprocess(args):
    for data_type in ['tr', 'cv']:
        for speaker in ['mix', 's1', 's2']:
            preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),  # 拼接路径
                               os.path.join(args.out_dir, data_type),
                               speaker,
                               sample_rate=args.sample_rate)
def mix(wav, index1):
    index2 = np.random.randint(0, len(wav))
    wav1, _ = librosa.load(wav[index1], sr=8000)
    wav1 = wav1 / np.max(wav1)
    wav2, _ = librosa.load(wav[index2], sr=8000)
    wav2 = wav2 / np.max(wav2)
    ratio = np.random.random() / 5 + 0.5
    if len(wav2) > len(wav1):
        mixture = wav1 + wav2[:len(wav1)] * ratio
    else:
        mixture = wav1.copy()
        mixture[:len(wav2)] += wav2 * ratio
    return wav1, wav2, mixture

def dataset(in_path, out_path1, out_path2):
    wav = []
    for f in os.listdir(in_path):
        wav.append(os.path.join(in_path, f))
    for i in range(len(wav)):
        if i > 20:
            out_path = out_path1
        else:
            out_path = out_path2
        wav1, wav2, mixture = mix(wav, i)
        sf.write(os.path.join(out_path, 's1', str(i) + '.wav'), wav1, 8000)
        sf.write(os.path.join(out_path, 's2', str(i) + '.wav'), wav2, 8000)
        sf.write(os.path.join(out_path, 'mix', str(i) + '.wav'), mixture, 8000)
    return wav

if __name__ == "__main__":

    # generate mix dataset
    #wav = dataset('bss/raw', 'bss/tr', 'bss/cv')

    parser = argparse.ArgumentParser("WSJ0 data preprocessing")

    parser.add_argument('--in-dir',
                        type=str,
                        default="./bss",
                        help='Directory path of wsj0 including tr, cv and tt')

    parser.add_argument('--out-dir',
                        type=str,
                        default="./json/",
                        help='Directory path to put output files')

    parser.add_argument('--sample-rate',
                        type=int,
                        default=8000,
                        help='Sample rate of audio file')

    args = parser.parse_args()

    preprocess(args)
