import torchaudio
import json
import os
import argparse

def load(path, files, audio=False):
    log = []
    for f in files:
        if audio:
            log.append([os.path.join(path, f), torchaudio.info(os.path.join(path, f)).num_frames])
        else:
            log.append([os.path.join(path, f), len(open(os.path.join(path, f), 'rb').readlines())])
    return log

def update(dict, name, file_list, N, p, kinds=4):
    file_list = sorted(file_list)
    N = int(N / kinds)
    imu = file_list[: 2 * N]
    gt = file_list[2 * N: 3 * N]
    wav = file_list[3 * N:]
    imu_files = load(path, imu, audio=False)
    gt_files = load(path, gt, audio=True)
    wav_files = load(path, wav, audio=True)
    if name in dict:
        if p in dict[name][0]:
            dict[name][0][p] += imu_files
            dict[name][1][p] += gt_files
            dict[name][2][p] += wav_files
        else:
            dict[name][0][p] = imu_files
            dict[name][1][p] = gt_files
            dict[name][2][p] = wav_files
    else:
        dict[name] = [{p: imu_files}, {p: gt_files}, {p: wav_files}]
    # if N % 4 == 0:
    #     N = int(N/4)
    #     # imu1 = file_list[: N]
    #     # imu2 = file_list[N: 2 * N]
    #     imu = file_list[: 2 * N]
    #     gt = file_list[2 * N: 3 * N]
    #     wav = file_list[3 * N:]
    #     imu_files = load(path, imu, audio=False)
    #     wav_files = load(path, wav, audio=True)
    #     gt_files = load(path, gt, audio=True)
    #     if name in dict:
    #         if p in dict[name][0]:
    #             dict[name][0][p] += imu_files
    #             dict[name][1][p] += wav_files
    #             dict[name][2][p] += gt_files
    #         else:
    #             dict[name][0][p] = imu_files
    #             dict[name][1][p] = wav_files
    #             dict[name][2][p] = gt_files
    #     else:
    #         dict[name] = [{p: imu_files}, {p: wav_files}, {p: gt_files}]
    # else:
    #     N = int(N / 3)
    #     # imu1 = file_list[: N]
    #     # imu2 = file_list[N: 2 * N]
    #     imu = file_list[: 2 * N]
    #     gt = file_list[2 * N: 3 * N]
    #     imu_files = load(path, imu, audio=False)
    #     gt_files = load(path, gt, audio=True)
    #     if name in dict:
    #         if p in dict[name][0]:
    #             dict[name][0][p] += imu_files
    #             dict[name][1][p] += gt_files
    #         else:
    #             dict[name][0][p] = imu_files
    #             dict[name][1][p] = gt_files
    #     else:
    #         dict[name] = [{p: imu_files}, {p: gt_files}]
    return dict
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-pre train, 1-main benchmark, 2-mirco benchmark')
    args = parser.parse_args()
    if args.mode == 0:
        directory = "../dataset/"
        datasets = ['dev', 'background', 'music', 'train']
        for dataset in datasets:
            audio_files = []
            g = os.walk(directory + dataset)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name[-3:] in ['wav', 'lac']:
                        audio_files.append([os.path.join(path, file_name), torchaudio.info(os.path.join(path, file_name)).num_frames])
            json.dump(audio_files, open('json/' + dataset + '.json', 'w'), indent=4)

        data = {}
        for dataset in ['dev', 'background', 'music']:
            with open('json/' + dataset + '.json', 'r') as f:
                data[dataset] = json.load(f)
        all_noise = []
        sum_time = {'dev':0, 'background':0, 'music':0}
        i = 0
        flag = 0
        while True:
            for dataset in ['dev', 'background', 'music']:
                if i >= len(data[dataset]):
                    flag = sum_time[dataset]
                else:
                    if flag > 0 and sum_time[dataset] > flag:
                        pass
                    else:
                        all_noise.append(data[dataset][i])
                        sum_time[dataset] += data[dataset][i][1]
            i = i + 1
            if i > 2700:
                break
        json.dump(all_noise, open('json/all_noise.json', 'w'), indent=4)
    elif args.mode == 1:
        directory = '../dataset/our'
        person = os.listdir(directory)
        dict = {}
        for p in person:
            g = os.walk(os.path.join(directory, p))
            for path, dir_list, file_list in g:
                N = len(file_list)
                if N > 0:
                    name = path.split('/')[-1]
                    dict = update(dict, name, file_list, N, p)
        for name in dict:
            # if len(dict[name]) == 2:
            #     json.dump(dict[name][0], open('json/' + name + '_imu.json', 'w'), indent=4)
            #     json.dump(dict[name][1], open('json/' + name + '_gt.json', 'w'), indent=4)
            # else:
                json.dump(dict[name][0], open('json/' + name + '_imu.json', 'w'), indent=4)
                json.dump(dict[name][1], open('json/' + name + '_gt.json', 'w'), indent=4)
                json.dump(dict[name][2], open('json/' + name + '_wav.json', 'w'), indent=4)
