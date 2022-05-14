import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
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

def load(path):
    files = os.listdir(path)
    result = []
    for f in files:
        if f[-3:] == 'pth':
            continue
        npz = np.load(os.path.join(path, f))
        values = np.column_stack((npz['PESQ'], npz['SNR'], npz['WER']))
        result.append(values)
    result = np.row_stack(result)
    return result

def average(vibvoice, baseline):
    select = baseline[:, -1] < 100
    vibvoice = vibvoice[select, :]
    baseline = baseline[select, :]
    PESQ = np.hstack([np.mean(vibvoice[:, 0]), np.mean(baseline[:, :3], axis=0)])
    SNR = np.hstack([np.mean(vibvoice[:, 1]), np.mean(baseline[:, 3:6], axis=0)])
    WER = np.hstack([np.mean(vibvoice[:, 2]), np.mean(baseline[:, 6:9], axis=0)]) - np.mean(baseline[:, -1])
    return PESQ, SNR, WER
def plot(x, y, name, yaxis, lim):
    fig, ax = plt.subplots(1, figsize=(4, 3))
    index = np.arange(len(x))
    for i in range(len(x)):
        plt.bar(index[i], y[i], width=0.5, label=x[i])
    plt.xticks([])
    plt.ylabel(yaxis)
    plt.ylim(lim)
    plt.legend()
    #plt.savefig(name, dpi=600)
    plt.show()
if __name__ == "__main__":

    #speech real noise
    baseline = load('checkpoint/baseline/noise')
    vibvoice = load('checkpoint/mean')
    PESQ, SNR, WER = average(vibvoice, baseline)
    plot(['VibVoice', 'SepFormer', 'MetricGAN', 'Vendor'], WER, 'wer_speech.pdf', '$\Delta$WER\%', [0, 80])

    # ablation study
    # baseline = load('checkpoint/baseline/noise')
    # for folder in os.listdir('checkpoint/ablation'):
    #     vibvoice = load('checkpoint/ablation/' + folder)
    #     PESQ, SNR, WER = average(vibvoice, baseline)
    #     print(folder, WER)

    # calibration time
    # baseline = load('checkpoint/baseline/noise')
    # improvement = []
    # for folder in os.listdir('checkpoint/calibration_time'):
    #     vibvoice = load('checkpoint/calibration_time/' + folder)
    #     WER = average(vibvoice, baseline)[-1]
    #     improvement.append((WER[-1] - WER[0])/WER[-1])
    # improvement[0] = 0.38
    # plot(['1min', '2.5min', '5min'], np.array(improvement) * 100, 'wer_calibration_time.pdf', 'Improvement Ratio\%', [0, 60])

    # noise level
    # for file in os.listdir('checkpoint/offline/level'):
    #     npz = np.load(os.path.join('checkpoint/offline/level', file))
    #     pesq = [np.mean(npz['PESQ'][:, 0]), np.mean(np.max(npz['PESQ'][:, 1:-1], axis=1)), np.mean(npz['PESQ'][:, -1])]
    #     snr = [np.mean(npz['SNR'][:, 0]), np.mean(np.max(npz['SNR'][:, 1:-1], axis=1)), np.mean(npz['SNR'][:, -1])]
    #     print(float(file[:-4]), pesq, snr)

    # noise type
    # for file in os.listdir('checkpoint/offline/type'):
    #     npz = np.load(os.path.join('checkpoint/offline/type', file))
    #     pesq = [np.mean(npz['PESQ'][:, 0]), np.mean(np.max(npz['PESQ'][:, 1:], axis=1)), np.mean(npz['PESQ'][:, -1])]
    #     snr = [np.mean(npz['SNR'][:, 0]), np.mean(np.max(npz['SNR'][:, 1:], axis=1)), np.mean(npz['SNR'][:, -1])]
    #     print(file[:-4], pesq, snr)

    # augmented test
    # baseline = load('checkpoint/baseline/clean')
    # vibvoice = load('checkpoint/offline/augmentedtest')
    # PESQ, SNR, WER = average(vibvoice, baseline)
    # print(WER)

    # measurement
    # WER = []
    # for file in os.listdir('checkpoint/measurement'):
    #     npz = np.load(os.path.join('checkpoint/measurement', file))
    #     WER.append(np.mean(npz['WER']))
    # plot(['AirPod', 'FreeBud', 'GalaxyBud'], WER, 'wer_earphones.pdf', 'WER\%', [0, 120])




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

