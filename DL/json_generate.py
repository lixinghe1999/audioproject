import torchaudio
import json
import os
import argparse

def load(path, files, audio=False):
    log = []
    if audio:
        for f in files:
            log.append([os.path.join(path, f), torchaudio.info(os.path.join(path, f)).num_frames])
    else:
        for f in files:
            log.append([os.path.join(path, f), len(open(os.path.join(path, f), 'rb').readlines())])
    return log

def update(dict, name, file_list, p, kinds=4):
    file_list = sorted(file_list)
    N = len(file_list)
    N = int(N / kinds)
    imu1 = file_list[: N]
    imu2 = file_list[N: 2 * N]
    gt = file_list[2 * N: 3 * N]
    wav = file_list[3 * N:]
    imu_files = load(path, imu1, audio=False) + load(path, imu2, audio=False)
    gt_files = load(path, gt, audio=True) + load(path, gt, audio=True)
    wav_files = load(path, wav, audio=True) + load(path, wav, audio=True)
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

    # imu1_files = load(path, imu1, audio=False)
    # imu2_files = load(path, imu2, audio=False)
    # gt_files = load(path, gt, audio=True)
    # wav_files = load(path, wav, audio=True)
    # if name in dict:
    #     if p in dict[name][0]:
    #         dict[name][0][p] += imu1_files
    #         dict[name][1][p] += imu2_files
    #         dict[name][2][p] += gt_files
    #         dict[name][3][p] += wav_files
    #     else:
    #         dict[name][0][p] = imu1_files
    #         dict[name][1][p] = imu2_files
    #         dict[name][2][p] = gt_files
    #         dict[name][3][p] = wav_files
    # else:
    #     dict[name] = [{p: imu1_files}, {p: imu2_files}, {p: gt_files}, {p: wav_files}]

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
                    # maybe different on Linux/ Windows
                    name = path.split('/')[-1]
                    if name in ['test', 'mask']:
                        dict = update(dict, name, file_list, p, kinds=3)
                    else:
                        dict = update(dict, name, file_list, p, kinds=4)
        for name in dict:
                json.dump(dict[name][0], open('json/' + name + '_imu.json', 'w'), indent=4)
                json.dump(dict[name][1], open('json/' + name + '_gt.json', 'w'), indent=4)
                json.dump(dict[name][2], open('json/' + name + '_wav.json', 'w'), indent=4)

                # json.dump(dict[name][0], open('json/' + name + '_imu1.json', 'w'), indent=4)
                # json.dump(dict[name][1], open('json/' + name + '_imu2.json', 'w'), indent=4)
                # json.dump(dict[name][2], open('json/' + name + '_gt.json', 'w'), indent=4)
                # json.dump(dict[name][3], open('json/' + name + '_wav.json', 'w'), indent=4)
