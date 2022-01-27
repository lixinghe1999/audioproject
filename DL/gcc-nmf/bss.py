from gccNMF.gccNMFFunctions import *
from gccNMF.gccNMFPlotting import *

windowSize = 1024
fftSize = windowSize
hopSize = 128
windowFunction = hanning
# TDOA params
numTDOAs = 128
# NMF params
dictionarySize = 128
numIterations = 100
sparsityAlpha = 0

# for example it is 1 and 3 data/dev1_female3_liverec_130ms_1m.wav
microphoneSeparationInMetres = 0.1
numSources = 2
def bss(mixtureFileName):
    # Input params
    stereoSamples, sampleRate = loadMixtureSignal(mixtureFileName)

    numChannels, numSamples = stereoSamples.shape
    durationInSeconds = numSamples / float(sampleRate)

    # figure(figsize=(14, 6))
    # plotMixtureSignal(stereoSamples, sampleRate)
    # plt.show()

    complexMixtureSpectrogram = computeComplexMixtureSpectrogram(stereoSamples, windowSize, hopSize, windowFunction)
    numChannels, numFrequencies, numTime = complexMixtureSpectrogram.shape
    frequenciesInHz = getFrequenciesInHz(sampleRate, numFrequencies)
    frequenciesInkHz = frequenciesInHz / 1000.0

    # figure(figsize=(12, 8))
    # plotMixtureSpectrograms(complexMixtureSpectrogram, frequenciesInkHz, durationInSeconds)
    # plt.show()

    spectralCoherenceV = complexMixtureSpectrogram[0] * complexMixtureSpectrogram[1].conj() / abs(complexMixtureSpectrogram[0]) / abs(complexMixtureSpectrogram[1])
    angularSpectrogram = getAngularSpectrogram(spectralCoherenceV, frequenciesInHz, microphoneSeparationInMetres, numTDOAs)
    meanAngularSpectrum = mean(angularSpectrogram, axis=-1)
    targetTDOAIndexes = estimateTargetTDOAIndexesFromAngularSpectrum(meanAngularSpectrum,microphoneSeparationInMetres,numTDOAs, numSources)

    # figure(figsize=(14, 6))
    # plotGCCPHATLocalization( spectralCoherenceV, angularSpectrogram, meanAngularSpectrum,
    #                          targetTDOAIndexes, microphoneSeparationInMetres, numTDOAs,
    #                          durationInSeconds )
    # plt.show()

    V = concatenate(abs(complexMixtureSpectrogram), axis=-1)
    W, H = performKLNMF(V, dictionarySize, numIterations, sparsityAlpha)
    numChannels = stereoSamples.shape[0]
    stereoH = array(hsplit(H, numChannels))
    targetTDOAGCCNMFs = getTargetTDOAGCCNMFs(spectralCoherenceV, microphoneSeparationInMetres,
                                              numTDOAs, frequenciesInHz, targetTDOAIndexes, W, stereoH)
    targetCoefficientMasks = getTargetCoefficientMasks(targetTDOAGCCNMFs, numSources)
    targetSpectrogramEstimates = getTargetSpectrogramEstimates(targetCoefficientMasks,
                                                                complexMixtureSpectrogram, W,
                                                                stereoH)
    targetSignalEstimates = getTargetSignalEstimates(targetSpectrogramEstimates, windowSize,
                                                      hopSize, windowFunction)
    saveTargetSignalEstimates(targetSignalEstimates, sampleRate, mixtureFileName[:-4])
if __name__ == "__main__":
    #bss('../../exp7/he_raw/s4/data/dev1_female3_liverec_130ms_1m.wav')
    #bss('data/dev1_female3_liverec_130ms_1m.wav')
    #bss('data/mic_1641387264.0301032.wav')
    bss('data/mic_1641018619.6465752.wav')
