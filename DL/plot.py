import matplotlib.pyplot as plt
import numpy as np
import os
def wer_process(x):
    x = x - x[:, 0][:, None]
    baseline = np.min(x[:, 2:4], axis=1)
    x = np.vstack([x[:, 1], baseline, x[:, 4]])
    return x.T

def pesq_process(x):
    baseline = np.max(x[:, 1:3], axis=1)
    x = np.vstack([x[:, 0], baseline, x[:, 3]])
    return x.T

def another_baseline(folder):
    files = os.listdir(folder)
    for i in range(len(files)):
        f = files[i]
        npz = np.load(os.path.join(folder, f))
        x1, x2 = npz['PESQ'], npz['WER'][:, 1] - npz['WER'][:, 0]
        if i == 0:
            PESQ = x1
            WER = x2
        else:
            PESQ = np.hstack([PESQ, x1])
            WER = np.hstack([WER, x2])
    metric = WER

    return metric
def load_baseline(path):
    files = os.listdir(path)
    baseline = []
    for f in files:
        npz = np.load(os.path.join(path, f))
        values = [np.mean(npz['PESQ'], axis=0), np.mean(npz['SNR'], axis=0), np.mean(npz['WER'], axis=0)]
        baseline.append(values)
    return baseline
def load_vibvoice(path):
    files = os.listdir(path)
    vibvoice = []
    for f in files:
        if f[-3:] == 'pth':
            continue
        npz = np.load(os.path.join(path, f))
        values = [npz['PESQ'], npz['SNR'], npz['WER']]
        vibvoice.append(values)
    return vibvoice

if __name__ == "__main__":
    baseline = load_baseline('checkpoint/baseline/noise')
    vibvoice = load_vibvoice('checkpoint/try')
    metric = []
    for i in range(15):
        print(baseline[i])
        print(vibvoice[i])
    folder = 'checkpoint/5min'
    files = os.listdir(folder)
    files = [f for f in files if f[-3:] == 'npz']
    for i in range(len(files)):
        f = files[i]
        npz = np.load(os.path.join(folder, f))
        x1, x2 = npz['PESQ'], npz['WER']
        x1 = pesq_process(x1)
        x2 = wer_process(x2)
        if i == 0:
            PESQ = x1
            WER = x2
        else:
            PESQ = np.vstack([PESQ, x1])
            WER = np.vstack([WER, x2])
    metric = WER

    # m = another_baseline('checkpoint/baseline/noise')
    # folder = 'checkpoint/5min'
    # files = os.listdir(folder)
    # files = [f for f in files if f[-3:] == 'npz']
    # for i in range(len(files)):
    #     f = files[i]
    #     npz = np.load(os.path.join(folder, f))
    #     x1, x2 = npz['PESQ'], npz['WER']
    #     x1 = pesq_process(x1)
    #     x2 = wer_process(x2)
    #     if i == 0:
    #         PESQ = x1
    #         WER = x2
    #     else:
    #         PESQ = np.vstack([PESQ, x1])
    #         WER = np.vstack([WER, x2])
    # metric = WER
    # metric = np.insert(metric, 2, m, axis=1)
    # print(metric.shape)
    # mean = np.mean(metric, axis=0)
    # var = np.std(metric, axis=0)
    # fig, ax = plt.subplots(1, figsize=(5, 4))
    # name = ['VibVoice', 'SepFormer', 'MetricGAN', 'vendor']
    # #c = ['b', 'r', 'y']
    # x = np.arange(len(name))
    # for i in range(len(x)):
    #     plt.bar(x[i], mean[i], yerr=var[i]/3, width=0.5, label=name[i])
    # plt.xticks([])
    # plt.ylabel(u'Δ WER/%')
    # #plt.ylabel('PESQ')
    # plt.legend()
    # plt.savefig('wer_all.eps', dpi=300)
    # plt.show()

    # folder = 'checkpoint/new'
    # files = os.listdir(folder)
    # files = [f for f in files if f[-3:] == 'npz']
    # ratio = []
    # for i in range(len(files)):
    #     f = files[i]
    #     npz = np.load(os.path.join(folder, f))
    #     x1, x2 = npz['PESQ'], npz['WER']
    #     x2 = wer_process(x2)
    #     ratio.append(np.mean(x2, axis=0))
    # x = np.arange(len(ratio))
    # ratio = np.array(ratio)
    # ratio = 100 * (ratio[:, -1] - ratio[:, 0]) / ratio[:, -1]
    # fig, ax = plt.subplots(1, figsize=(5, 4))
    # plt.bar(x, ratio)
    # plt.xticks([])
    # plt.ylabel('Ratio/%')
    # plt.savefig('wer_each.eps', dpi=300)
    # plt.show()

    # folder = 'checkpoint/earphones'
    # files = os.listdir(folder)
    # files = [f for f in files if f[-3:] == 'npz']
    # ratio = []
    # for i in range(len(files)):
    #     f = files[i]
    #     npz = np.load(os.path.join(folder, f))
    #     x1, x2 = npz['PESQ'], npz['WER']
    #     x2 = wer_process(x2)
    #     ratio.append(np.mean(x2, axis=0))
    # x = np.arange(len(ratio))
    # ratio = np.array(ratio)
    # print(ratio)
    # ratio1 = (ratio[:, -1] - ratio[:, 0]) / ratio[:, -1] * 100
    # fig, ax = plt.subplots(1, figsize=(5, 4))
    # name = ['airpods', 'freebud', 'galaxybud']
    # c = ['b', 'r', 'y']
    # width = 0.3
    # for i in range(len(x)):
    #     plt.bar(x[i], ratio1[i], width=width, label=name[i], color=[c[i]])
    #     #plt.bar(x[i] + 1.5 * width, ratio2[i], width=width, label=name[i], color=[c[i]])
    # plt.xticks([])
    # plt.ylabel('Ratio/%')
    # plt.legend()
    # plt.savefig('wer_earphones.eps', dpi=300)
    # plt.show()

    # folder = 'checkpoint/synthetic'
    # files = os.listdir(folder)
    # WER = []
    # VAR = []
    # N = int(len(files)/2)
    # for i in range(N):
    #     f = files[i]
    #     npz1 = np.load(os.path.join(folder, f), allow_pickle=True)
    #     f = files[N + i]
    #     npz2 = np.load(os.path.join(folder, f), allow_pickle=True)
    #     x = np.vstack([npz1['WER'], npz2['WER']])
    #     x = wer_process(x)
    #     VAR.append(np.std(x[:, 0]))
    #     x = np.mean(x, axis=0)
    #     WER.append(100 * (x[-1] - x[0])/x[-1])
    # fig, ax = plt.subplots(1, figsize=(5, 4))
    # x = np.arange(N)
    # print(WER, VAR)
    # width = 0.3
    # plt.bar(x, WER, yerr=VAR, width=width, color='b')
    # plt.xticks(x, ['still', 'move'])
    # plt.ylabel('Ratio/%')
    # plt.legend()
    # plt.savefig('wer_noise.eps', dpi=300)
    # plt.show()

    # folders = ['new', '5min', '2.5min', '1min', 'nopretrain']
    # plt_ratio = []
    # plt_var = []
    # for folder in folders:
    #     files = os.listdir('checkpoint/' + folder)
    #     files = [f for f in files if f[-3:] == 'npz']
    #     ratio = []
    #     for i in range(len(files)):
    #         f = files[i]
    #         npz = np.load(os.path.join('checkpoint/', folder, f))
    #         x1, x2 = npz['PESQ'], npz['WER']
    #         x2 = wer_process(x2)
    #         ratio.append(np.mean(x2, axis=0))
    #     ratio = np.array(ratio)
    #     ratio = 100 * (ratio[:, -1] - ratio[:, 0]) / ratio[:, -1]
    #     plt_ratio.append(np.mean(ratio))
    #     plt_var.append(np.std(ratio))
    # x = np.arange(len(folders))
    # width = 0.3
    # fig, ax = plt.subplots(1, figsize=(5, 4))
    # plt.bar(x, plt_ratio, width=width)
    # plt.xticks(x, folders)
    # plt.ylabel('Ratio/%')
    # plt.savefig('wer_calibration.eps', dpi=300)
    # plt.show()

    # folder = 'checkpoint/field'
    # files = os.listdir(folder)
    # files = [f for f in files if f[-3:] == 'npz']
    # WER = []
    # for i in range(len(files)):
    #     f = files[i]
    #     npz = np.load(os.path.join(folder, f))
    #     x1, x2 = npz['PESQ'], npz['WER']
    #     x2 = wer_process(x2)
    #     select = x2[:, -1] > 0
    #     a = np.mean(x2[select, :], axis=0)
    #     WER.append((a[-1] - a[0])/a[-1])
    # print(WER)
    # fig, ax = plt.subplots(1, figsize=(5, 4))
    # name = ['corridor', 'meeting', 'office', 'stair']
    # x = np.arange(len(name))
    # for i in range(len(x)):
    #     plt.bar(x[i], WER[i], width=0.5, label=name[i])
    # plt.xticks([])
    # plt.ylabel(u'Δ WER/%')
    # plt.legend()
    # plt.savefig('wer_field.eps', dpi=300)
    # plt.show()

