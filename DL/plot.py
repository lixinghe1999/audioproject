import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    # this script will plot the result
    # result -- noise [0.5, 0.7]
    # hou
    # PESQ = [2.49791462, 2.44536132, 1.63476246, 1.44009903, 1.62870837]
    # WER = [64.56685499, 86.77024482, 88.01318267, 111.65725047, 122.38229755, 132.13747646]
    # he
    # PESQ = [2.32890369, 2.31776339, 1.66573004, 1.47485123, 1.68319492]
    # WER = [47.42770167, 96.59817352, 91.45357686, 114.33789954, 130.08371385, 140.2891933 ]

    # real noise

    # shuai
    # PESQ = [1.30926762, 1.30350545, 1.30646505, 1.33991065, 1.29739612]
    # WER = [ 57.22222222, 105.77777778, 105, 114.38888889, 118.61111111, 133.66666667]
    # shi
    # PESQ = [1.65899728 1.63522623 1.67117212 1.60802434 1.56386758]
    # WER = [ 55.13888889,  68.27777778,  65.58333333,  68.77777778, 101.16666667, 111.44444444]
    # he
    # PESQ = [1.32444969, 1.34057104, 1.49953943, 1.34908197, 1.37570135]
    # WER = [ 66.77777778, 128.91666667, 105.13888889,  82.38888889, 129.22222222, 131.72222222]
    width = 0.2
    name = ["vanilla", 'extra', 'single', 'baseline', 'original']
    WER = WER[:-1]
    x = np.arange(len(name))
    fig, axs = plt.subplots(2, 1)
    axs[0].bar(x, PESQ)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(name)
    axs[1].bar(x, WER)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(name)
    axs[0].set_ylabel('PESQ')
    axs[1].set_ylabel('WER/%')
    plt.savefig('wer_pesq.eps', dpi=300)
