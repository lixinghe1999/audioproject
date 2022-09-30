import torch
from torch.nn import functional

from audio_zen.acoustics.feature import drop_band
from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel
import time
from torchvision.utils import save_image
from torch.utils.mobile_optimizer import optimize_for_mobile

class Sequence_A2net(BaseModel):
    def __init__(self,
                 audio_freqs=297,
                 acc_freqs=264,
                 look_ahead=2,
                 sequence_model="LSTM",
                 fb_output_activate_function="ReLU",
                 fb_model_hidden_size=512,
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 weight_init=True,
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
        self.acc_freqs = acc_freqs
        self.audio_freqs = audio_freqs
        self.fb_model = SequenceModel(
            input_size=self.acc_freqs + self.audio_freqs,
            output_size=self.acc_freqs + self.audio_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )


        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag, acc):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram
            acc: corresponding accelerometer reading

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            acc: [B, 1, F_a, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        noisy_mag = torch.cat([noisy_mag, acc], dim=2)
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead]) # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Fullband model

        fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)
        output = fb_output[:, :, :, self.look_ahead:]
        recon_audio = output[:, :, :self.audio_freqs, :]
        recon_acc = output[:, :, -self.acc_freqs:, :]
        return recon_audio, recon_acc

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

    audio = torch.rand(2, 1, 264, 151)
    acc = torch.rand(2, 1, 33, 151)
    model = Sequence_A2net()
    output = model(audio, acc)
    print(output.shape)
    size_all_mb = model_size(model)
    print('model size: {:.3f}MB'.format(size_all_mb))
    #
    latency = model_speed(model, [audio,acc])
    print('model latency: {:.3f}S'.format(latency))

    #model_save(model, audio)