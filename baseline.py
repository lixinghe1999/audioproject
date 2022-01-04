import librosa
import matplotlib.pyplot as plt
import numpy as np
from speechbrain.pretrained import EncoderDecoderASR, SpectralMaskEnhancement
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
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


    asr_model = EncoderDecoderASR.from_hparams(source="pretrained_models/asr-transformer-transformerlm-librispeech",
                                               savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
                                               run_opts={"device":"cuda"})

    audio_files = []
    audio_files.append('test/clip1.wav')
    audio_files.append('test/clip2.wav')
    audio_files.append('test/clip7.wav')
    audio_files.append('test/clip3.wav')
    audio_files.append('test/clip4.wav')
    audio_files.append('test/clip5.wav')
    audio_files.append('test/clip6.wav')
    #audio_files.append('exp6\he\clean\mic_1639205708.8325815.wav')
    #audio_files.append('exp6\he\clean\mic_1639205761.9480312.wav')
    sigs = []
    lens = []
    for audio_file in audio_files:
        #snt, fs = librosa.load(audio_file, mono=False)
        snt, fs = torchaudio.load(audio_file)
        sigs.append(snt[0])
        lens.append(snt.shape[1])
    batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)
    lens = torch.Tensor(lens) / batch.shape[1]
    print(asr_model.transcribe_batch(batch, lens))

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