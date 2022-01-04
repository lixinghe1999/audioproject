import librosa
from pesq import pesq
import micplot
import numpy as np
import sys

def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.

    Main algorithm used is dynamic programming.

    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.

    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y - 1] + 1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + 1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]


def alignedPrint(list, r, h, result):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.

    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
    print("\nWER: " + result)


def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    # build the matrix
    d = editDistance(r, h)

    # find out the manipulation steps
    list = getStepList(r, h, d)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    #result = str("%.2f" % result) + "%"
    #alignedPrint(list, r, h, result)
    return result


## we evaluate WER and PESQ in this script
if __name__ == "__main__":
    # filename1 = sys.argv[1]
    # filename2 = sys.argv[2]
    # with open(filename1, 'r', encoding="utf8") as ref:
    #     r = ref.read().split()
    # with open(filename2, 'r', encoding="utf8") as hyp:
    #     h = hyp.read().split()
    # wer(r, h)
    #
    reference = 'demo/clean.wav'
    wave_clean, _ = librosa.load(reference, sr=16000)
    noise = 'conversation.wav'
    wave_noise, _ = librosa.load(noise, sr=16000)
    noisy1 = micplot.add_noise(wave_noise, wave_clean, ratio=0.1)
    noisy2 = micplot.add_noise(wave_noise, wave_clean, ratio=0.05)
    score = pesq(16000, wave_clean, noisy1, 'wb')
    print(score)
    score = pesq(16000, wave_clean, noisy2, 'wb')
    print(score)
