import torch
from torch.nn import functional
import torch.nn as nn
from audio_zen.acoustics.feature import drop_band
from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel
import time
from torchvision.utils import save_image
from torch.utils.mobile_optimizer import optimize_for_mobile

class Unet_encoder(nn.Module):
    def __init__(self, filters=[16, 32, 64, 128], kernels=[[3, 3], [3, 3], [3, 3], [3, 3]], max_pooling={2:[1,3], 3:[2,1]}):
        super(Unet_encoder, self).__init__()
        self.num_layers = len(filters)
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_channel = 1
            else:
                input_channel = filters[i-1]
            output_channel = filters[i]
            kernel = kernels[i]
            padding = [(k-1)//2 for k in kernel]
            layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True))
            layers.append(layer)
            if i in max_pooling:
                layers.append(nn.MaxPool2d(kernel_size=max_pooling[i]))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
class Unet_decoder(nn.Module):
    def __init__(self, input_channel, filters=[16, 32, 64, 128],
                 kernels=[[3, 3], [3, 3], [3, 3], [3, 3]], max_pooling={2:[1,3], 3:[2,1]}):
        super(Unet_decoder, self).__init__()
        self.num_layers = len(filters)
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_channel = input_channel
            else:
                input_channel = filters[i-1]
            output_channel = filters[i]
            kernel = kernels[i]
            padding = [(k-1)//2 for k in kernel]
            layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True))
            layers.append(layer)
            if i in max_pooling:
                layers.append(nn.ConvTranspose2d(output_channel, output_channel, kernel_size=max_pooling[i], stride=max_pooling[i]))
        layers.append(nn.Conv2d(filters[-1], 1, kernel_size=1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
class Sequence_A2net(BaseModel):
    def __init__(self,
                 T_segment=25,
                 sequence_model="LSTM",
                 fb_output_activate_function="ReLU",
                 ):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            look_ahead: Number of use of the future frames
            fb_num_neighbors: How much neighbor frequencies at each side from fullband model's output
            sb_num_neighbors: How much neighbor frequencies at each side from noisy spectrogram
            sequence_model: Chose one sequence model as the basic model e.g., GRU, LSTM
            fb_output_activate_function: fullband model's activation function
            sb_output_activate_function: subband model's activation function
            norm_type: type of normalization, see more details in "BaseModel" class
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."
        self.T_segment = T_segment
        self.model_acc = Unet_encoder(filters=[16, 32, 64], kernels=[[3, 3], [3, 3], [3, 3]],
                                 max_pooling={2: [3, 1]})
        self.model_audio = Unet_encoder(filters=[16, 32, 64, 128],
                                   kernels=[[5, 5], [5, 5], [5, 5], [3, 3]],
                                   max_pooling={0: [2, 1], 1: [2, 1], 2: [2, 1], 3: [3, 1]})
        self.model_fusion = Unet_decoder(input_channel=128+64, filters=[128, 64, 32, 16],
                                   kernels=[[5, 5], [5, 5], [5, 5], [3, 3]],
                                   max_pooling={0: [2, 1], 1: [2, 1], 2: [2, 1], 3: [3, 1]})
        self.fb_model = SequenceModel(
            input_size=(128 + 64) * 11,
            output_size=512,
            hidden_size=512,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )
        self.fc = nn.Sequential(nn.Linear(512, 400), nn.Linear(400, 264))

    def forward(self, acc, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram
            acc: corresponding accelerometer reading

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            acc: [B, 1, F_a, T]
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        noisy_mag = noisy_mag.reshape(-1, num_channels, num_freqs, self.T_segment)
        noisy_mag = self.model_audio(noisy_mag)
        noisy_mag = noisy_mag.reshape(batch_size, 128, 11, -1)

        batch_size, num_channels, num_freqs, num_frames = acc.size()
        acc = acc.reshape(-1, num_channels, num_freqs, self.T_segment)
        acc = self.model_acc(acc)
        acc = acc.reshape(batch_size, 64, 11, -1)

        noisy_mag = torch.cat([noisy_mag, acc], dim=1)
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        # Fullband model
        noisy_mag = noisy_mag.reshape(batch_size, num_channels * num_freqs, num_frames)
        output = self.fb_model(noisy_mag)
        output = self.fc(output.permute(0, 2, 1)).permute(0, 2, 1)
        # output = self.fb_model(noisy_mag).reshape(batch_size, num_channels, num_freqs, num_frames)
        # output = self.model_fusion(output)
        return output

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2

    return size_all_mb
def model_speed(model, input):
    t_start = time.time()
    step = 20
    with torch.no_grad():
        for i in range(step):
            model(*input)
    return (time.time() - t_start)/step
def model_save(model, audio):
    from torch.jit.mobile import (
                _backport_for_mobile,
                _get_model_bytecode_version,
            )
    model.eval()
    scripted_module = torch.jit.trace(model, audio)
    scripted_module.save("inference.pt")
    optimized_scripted_module = optimize_for_mobile(scripted_module)
    optimized_scripted_module._save_for_lite_interpreter("inference_optimized.ptl")
    print("model version", _get_model_bytecode_version(f_input="inference.ptl"))
    save_image(audio, 'input.jpg')
if __name__ == "__main__":

    audio = torch.rand(2, 1, 264, 150)
    acc = torch.rand(2, 1, 33, 150)
    # model_acc = Unet_encoder(filters=[16, 32, 64, 128], kernels=[[3, 3], [3, 3], [3, 3], [3, 3]], max_pooling={2:(1, 2), 3:(3,1)})
    # model_audio = Unet_encoder(filters=[16, 32, 64, 128, 256],
    #                            kernels=[[5, 5], [5, 5], [5, 5], [5, 5], [3, 3]], max_pooling={1: [2, 1], 2: [2, 1], 3: [2, 1],
                                                                                              #4: [3, 2]})

    model = Sequence_A2net()
    output = model(acc, audio)
    size_all_mb = model_size(model)
    print('model size: {:.3f}MB'.format(size_all_mb))
    latency = model_speed(model, [acc, audio])
    print('model latency: {:.3f}S'.format(latency))

    #model_save(model, audio)