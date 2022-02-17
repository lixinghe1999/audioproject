import librosa
import matplotlib.pyplot as plt
import numpy as np
from speechbrain.pretrained import EncoderDecoderASR, SpectralMaskEnhancement
from speechbrain.pretrained import SepformerSeparation as separator
from DL.evaluation import wer
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import soundfile as sf
# This is baseline for 1. Noise suppression 2. beamforming: webrtc, coherence based 3. facebook denoiser 4. Speech brain
# basically this script is only for checking
if __name__ == "__main__":
    # Noise Suppression
    # Linux: python3 test.py in C://Users/HeLix/py-webrtcns
    # sample rate 16000, 1 channel

    # Beamforming webrtc
    # Linux: ./webrtc-bf -i input.wav -mic_positions "x1 y1 z1 x2 y2 z2" -o output.wav in C://Users/HeLix/webrtc-beamforming/
    # multiple channels

    # Beamforming coherence based
    # Windows: applyProcess_real.m 3 types of coherence function, in D://audioproject/coherence/

    # facebook denoiser
    # Windows python -m denoiser.enhance --model_path=<path to the model> --noisy_dir=<path to the dir with the noisy files>
    # --out_dir=<path to store enhanced files>
    # --dns48, --dns64 or --master64

    # speech brain just in this script
    # below is ASR, trained on Librispeech, available for batch inference, need admin for cache


    # asr_model = EncoderDecoderASR.from_hparams(source="pretrained_models/asr-transformer-transformerlm-librispeech",
    #                                            savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
    #                                            run_opts={"device":"cuda"})
    #
    # text1 = asr_model.transcribe_file(r'exp7/hou/s1/noise/新录音 312.wav').split()
    # text2 = asr_model.transcribe_file(r'exp7/airpod/s1/新录音 312.wav').split()
    # gt = ["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"]
    # print(wer(text1, gt))
    # print(wer(text2, gt))

    #
    # audio_1 = 'D://audioproject\exp6\he\clean\mic_1639205708.8325815.wav'
    # asr_model.transcribe_file(audio_1)

    #below is speech enhancement, dataset voicebank

    # enhance_model = SpectralMaskEnhancement.from_hparams(source="pretrained_models/metricgan-plus-voicebank", run_opts={"device":"cuda"})
    # # Load and add fake batch dimension
    # audio_1 = 'demo/noisy1.wav'
    # noisy, fs = torchaudio.load(audio_1)
    # # Add relative length tensor
    # enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
    # # Saving enhanced signal on disk
    # torchaudio.save('enhanced.wav', enhanced.cpu(), 16000, bits_per_sample = 16)

    ## blind separation
    # def convergence_callback(Y):
    #     global error
    #     y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
    #     y = y[L - hop:, :].T
    #     m = np.minimum(y.shape[1], ref.shape[1])
    #     mae = np.abs(ref[:, :m], y[:, :m]).mean()
    #     error.append(mae)
    # def blind_separation(noise, y):
    #     L = 640
    #     hop = 320
    #     win_a = pra.hamming(L)
    #     win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)
    #     print(noise.T.shape)
    #     X = pra.transform.stft.analysis(noise.T, L, hop, win=win_a)
    #     print(X.shape)
    #     ref = y
    #     mae = np.abs(ref - noise).mean()
    #     print(mae)
    #
    #     Y = pra.bss.auxiva(X, n_iter=30, proj_back=True, callback=convergence_callback)
    #     y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
    #     y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
    #     y = y[L - hop:, :].T
    #     m = np.minimum(y.shape[1], ref.shape[1])
    #
    #     mae = np.abs(ref - y).mean()
    #     return mae

    model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr', run_opts={"device":"cuda"})
    # for custom file, change path
    est_sources = model.separate_file(path='test.wav')
    torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)

    asr_model = EncoderDecoderASR.from_hparams(source="pretrained_models/asr-transformer-transformerlm-librispeech",
                                               savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
                                               run_opts={"device":"cuda"})
    text1 = asr_model.transcribe_file("source1hat.wav").split()
    text2 = asr_model.transcribe_file("source2hat.wav").split()
    print(text1)
    print(text2)