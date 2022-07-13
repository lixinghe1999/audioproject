import torchaudio
import json
import os


def update(dict, name, file_list, N, p):
    imu1 = file_list[: N]
    imu2 = file_list[N: 2 * N]
    gt = file_list[2 * N: 3 * N]
    wav = file_list[3 * N:]
    imu_files = []
    wav_files = []
    gt_files = []
    for i in range(N):
        imu_files.append([os.path.join(path, imu1[i]), len(open(os.path.join(path, imu1[i])).readlines())])
        wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
        gt_files.append([os.path.join(path, gt[i]), torchaudio.info(os.path.join(path, gt[i])).num_frames])

        imu_files.append([os.path.join(path, imu2[i]), len(open(os.path.join(path, imu2[i])).readlines())])
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
    mode = 1
    if mode == 0:
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
    elif mode == 1:
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


    # elif mode == 2:
    #     # field = ['office', 'corridor', 'stair']
    #     # norm('field_paras.pkl', ['field_imuexp7.json', 'field_gtexp7.json', 'field_wavexp7.json'], False, field)
    #     # norm('field_train_paras.pkl', ['field_train_imuexp7.json', 'field_train_gtexp7.json', 'field_train_wavexp7.json'],False, ['canteen', 'station'])
    #
    #     candidate_all = ['yan', 'he', 'hou', 'shi', 'shuai', 'wu', 'liang', "1", "2", "3", "4", "5", "6", "7", "8", "airpod", "galaxy", 'freebud']
    #
    #     candidate = ['yan', 'he', 'hou', 'shi', 'shuai', 'wu', 'liang', "1", "2", "3", "4", "5", "6", "7", "8"]
    #
    #     norm('noise_train_paras.pkl', ['json/noise_train_imuexp7.json', 'json/noise_train_gtexp7.json', 'json/noise_train_wavexp7.json'], False, candidate_all)
    #
    #     norm('clean_train_paras.pkl', ['json/clean_train_imuexp7.json', 'json/clean_train_wavexp7.json', 'json/clean_train_wavexp7.json'], True, candidate)
    #
    #     norm('noise_paras.pkl', ['json/noise_imuexp7.json', 'json/noise_gtexp7.json', 'json/noise_wavexp7.json'], False, candidate_all)
    #
    #     norm('clean_paras.pkl', ['json/clean_imuexp7.json', 'json/clean_wavexp7.json', 'json/clean_wavexp7.json'], True, ['he', 'hou'])
    #
    #     norm('mobile_paras.pkl', ['json/mobile_imuexp7.json', 'json/mobile_wavexp7.json', 'json/mobile_wavexp7.json'], True, ['he', 'hou'])
    #
    # else:
    #     transfer_function, variance = read_transfer_function('../transfer_function')
    #     for i in range(19):
    #         index = np.random.randint(0, N)
    #         plt.plot(transfer_function[index, :])
    #         plt.show()
