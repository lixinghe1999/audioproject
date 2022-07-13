import torchaudio
import json
import os
import argparse


def update(dict, name, file_list, N, p):
    imu1 = file_list[: N]
    imu2 = file_list[N: 2 * N]
    gt = file_list[2 * N: 3 * N]
    wav = file_list[3 * N:]
    imu_files = []
    wav_files = []
    gt_files = []
    for i in range(N):
        imu_files.append([os.path.join(path, imu1[i]), len(open(os.path.join(path, imu1[i]), 'rb').readlines())])
        wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
        gt_files.append([os.path.join(path, gt[i]), torchaudio.info(os.path.join(path, gt[i])).num_frames])

        imu_files.append([os.path.join(path, imu2[i]), len(open(os.path.join(path, imu2[i]), 'rb').readlines())])
        wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
        gt_files.append([os.path.join(path, gt[i]), torchaudio.info(os.path.join(path, gt[i])).num_frames])
    if name in dict:
        if p in dict[name][0]:
            dict[name][0][p] += imu_files
            dict[name][1][p] += wav_files
            dict[name][2][p] += gt_files
        else:
            dict[name][0][p] = imu_files
            dict[name][1][p] = wav_files
            dict[name][2][p] = gt_files
    else:
        dict[name] = [{p: imu_files}, {p: wav_files}, {p: gt_files}]
    return dict
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-pre train, 1-main benchmark, 2-mirco benchmark')
    args = parser.parse_args()
    if args.mode == 0:
        directory = "../dataset/"
        datasets = ['dev', 'background', 'music', 'train']
        #datasets = ['dev']
        for dataset in datasets:
            audio_files = []
            g = os.walk(directory + dataset)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name[-3:] in ['wav', 'lac']:
                        audio_files.append([os.path.join(path, file_name), torchaudio.info(os.path.join(path, file_name)).num_frames])
            json.dump(audio_files, open('json/' + dataset + '.json', 'w'), indent=4)
    elif args.mode == 1:
        directory = '../dataset/our'
        person = os.listdir(directory)
        dict = {}
        for p in person:
            g = os.walk(os.path.join(directory, p))
            for path, dir_list, file_list in g:
                N = int(len(file_list) / 4)
                if N > 0:
                    name = path.split('\\')[-1]
                    dict = update(dict, name, file_list, N, p)
        for name in dict:
            json.dump(dict[name][0], open('json/' + name + '_imu.json', 'w'), indent=4)
            json.dump(dict[name][1], open('json/' + name + '_wav.json', 'w'), indent=4)
            json.dump(dict[name][2], open('json/' + name + '_gt.json', 'w'), indent=4)