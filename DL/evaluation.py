
import numpy as np


sentences = [["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"],
                    ["WE", "WANT", "TO", "IMPROVE", "SPEECH", "QUALITY", "IN", "THIS", "PROJECT"],
                    ["BUT", "WE", "DON'T", "HAVE", "ENOUGH", "DATA", "TO", "TRAIN", "OUR", "MODEL"],
                    ["TRANSFER", "FUNCTION", "CAN", "BE", "A", "GOOD", "HELPER", "TO", "GENERATE", "DATA"]]
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
    return float(d[len(r)][len(h)])

def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    # build the matrix
    d = editDistance(r, h)
    # print the result in aligned way
    result = d / len(r) * 100
    #result = d / (d + c) * 100
    #result = str("%.2f" % result) + "%"
    #alignedPrint(list, r, h, result)
    return result

def snr(gt, est):
    n = gt - est
    power_n = np.mean(np.abs(n)**2)
    power_gt = np.mean(np.abs(gt) ** 2)
    return 10*np.log10(power_gt/power_n)
## we evaluate WER and PESQ in this script
if __name__ == "__main__":
    # r = ['we', 'like', 'python']
    # h = ['wu', 'like', 'cython', 'but', 'we', 'just', 'want', 'to', 'but', 'we', 'just', 'want', 'to']
    # print(wer(r, h))
    f = open('survey.txt', 'r',encoding='UTF-8')
    lines = f.readlines()
    WER = []
    for i in range(len(lines)):
        hy = lines[i].upper().split()
        if len(hy) < 3:
            print(hy)
            continue
        gt = sentences[i // int(len(lines) / len(sentences))]
        WER.append(wer(gt, hy))
    print(np.mean(WER))
    # ground_truth = "hello world"
    # hypothesis = "hello duck see what happen so I like it that"
    #
    # wer = jiwer.wer(ground_truth, hypothesis)
    # mer = jiwer.mer(ground_truth, hypothesis)
    # wil = jiwer.wil(ground_truth, hypothesis)
    # print(wer, mer, wil)

