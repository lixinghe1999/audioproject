import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
from torch.nn.modules.utils import _pair

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        if padding is None:
            padding = [int((kernel_size[i] -1) * dilation[i]) for i in range(len(kernel_size))]
        self.left_padding = _pair(padding)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation,
                                           groups=groups, bias=bias)
    def forward(self, inputs):
        inputs = F.pad(inputs, (self.left_padding[1], 0, self.left_padding[0], 0))
        output = super().forward(inputs)
        return output
    
class CausalTransConv2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        if padding is None:
            padding = [int((dilation[i] * (kernel_size[i] - 1) + 1 - stride[i])) for i in range(len(kernel_size))]
        self.left_padding = _pair(padding)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation,
                                           groups=groups, bias=bias)
    def forward(self, inputs):
        inputs = F.pad(inputs, (self.left_padding[1], 0, self.left_padding[0], 0))
        output = super().forward(inputs)
        return output

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(in_channels,out_channels,kernel_size, 
                                           stride=stride,padding=0,dilation=dilation, groups=groups, bias=bias)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(CausalConv1d, self).forward(x)
class IMU_branch(nn.Module):
    def __init__(self, inference):
        super(IMU_branch, self).__init__()
        self.conv1 = nn.Sequential(
            CausalConv2d(1, 16, kernel_size=3, stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            CausalConv2d(16, 32, kernel_size=5, dilation=2, stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            CausalConv2d(32, 64, kernel_size=5, dilation=2, stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            CausalConv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.inference = inference
        if not self.inference:
            self.conv5 = nn.Sequential(
                CausalConv2d(128, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            self.conv6 = nn.Sequential(
                CausalTransConv2d(64, 32, kernel_size=(2, 1), stride=(2, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True))
            self.conv7 = nn.Sequential(
                CausalTransConv2d(32, 16, kernel_size=(2, 1), stride=(2, 1)),
                CausalConv2d(16, 1, kernel_size=3),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True))

    def forward(self, x):
        # down-sample
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_embedding = self.conv4(x)
        if self.inference:
            return x_embedding
        else:
            x = self.conv5(x_embedding)
            x = self.conv6(x)
            x = self.conv7(x)
            return x_embedding, x
class Audio_branch(nn.Module):
    def __init__(self):
        super(Audio_branch, self).__init__()
        self.conv1 = nn.Sequential(
            CausalConv2d(1, 16, kernel_size=5, stride=(4, 1), dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            CausalConv2d(16, 32, kernel_size=5, stride=(2, 1), dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )
        self.conv3 = nn.Sequential(
            CausalConv2d(32, 64, kernel_size=5, stride=(2, 1), dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.conv4 = nn.Sequential(
            CausalConv2d(64, 128, kernel_size=3, stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
        self.conv5 = nn.Sequential(
            CausalConv2d(128, 256, kernel_size=3, stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return [x1, x2, x3, x4, x5]

class Residual_branch(nn.Module):
    def __init__(self, in_channels):
        super(Residual_branch, self).__init__()
        self.r1 = nn.Sequential(
            CausalTransConv2d(in_channels, 128, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.r2 = nn.Sequential(
            CausalTransConv2d(128+128, 64, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.r3 = nn.Sequential(
            CausalTransConv2d(64+64, 32, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.r4 = nn.Sequential(
            CausalTransConv2d(32+32, 16, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.final = nn.Sequential(
            CausalTransConv2d(16+16, 16, kernel_size=(4, 1), stride=(4, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            CausalConv2d(16, 1, kernel_size=1))
    def forward(self, x, Res_x):
        x1, x2, x3, x4 = Res_x
        x = torch.cat([self.r1(x), x4], dim=1)
        x = torch.cat([self.r2(x), x3], dim=1)
        x = torch.cat([self.r3(x), x2], dim=1)
        x = torch.cat([self.r4(x), x1], dim=1)

        return self.final(x)


class Causal_A2net(nn.Module):
    def __init__(self, inference=False):
        super(Causal_A2net, self).__init__()
        self.inference = inference
        self.IMU_branch = IMU_branch(self.inference)
        self.lstm_layer = nn.LSTM(input_size=1536, hidden_size=1024, num_layers=1, batch_first=True)
        self.Audio_branch = Audio_branch()
        self.Residual_branch = Residual_branch(256)
        #self.Fusion_branch = Fusion_branch(256)
    def forward(self, acc, audio):
        if self.inference:
            acc = self.IMU_branch(acc)
            [x1, x2, x3, x4, x5] = self.Audio_branch(audio)
            x = torch.cat([acc, x5], dim=1)
            batch_size, n_channels, n_f_bins, n_frame_size = x.shape
            # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
            lstm_in = x.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
            lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
            x = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]
            x = self.Residual_branch(x, [x1, x2, x3, x4]) * audio
            #x = self.Fusion_branch(x) * x2
            return x
        else:
            acc = self.IMU_branch(acc)
            acc, acc_extra = acc
            [x1, x2, x3, x4, x5] = self.Audio_branch(audio)
            x = torch.cat([acc, x5], dim=1)

            batch_size, n_channels, n_f_bins, n_frame_size = x.shape
            lstm_in = x.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
            x, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
            x = x.permute(0, 2, 1).reshape(batch_size, -1, n_f_bins, n_frame_size)

            x = self.Residual_branch(x, [x1, x2, x3, x4]) * audio
            return x, acc_extra



def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

def model_save(model):
    model.eval()
    x = torch.rand((1, 1, 33, 151))
    noise = torch.rand((1, 1, 264, 151))
    scripted_module = torch.jit.trace(model, [x, noise])
    #scripted_module.save("inference.pt")
    optimized_scripted_module = optimize_for_mobile(scripted_module)
    optimized_scripted_module._save_for_lite_interpreter("inference.ptl")

    save_image(x, 'input1.jpg')
    save_image(noise, 'input2.jpg')

def model_speed(model, input):
    t_start = time.time()
    step = 100
    with torch.no_grad():
        for i in range(step):
            model(*input)
    return (time.time() - t_start)/step
if __name__ == "__main__":

    acc = torch.rand(1, 1, 32, 250)
    audio = torch.rand(1, 1, 256, 250)
    model = Causal_A2net(inference=False)
    recover_audio, recover_acc = model(acc, audio)
    print(recover_audio.shape, recover_acc.shape)
    print(model_size(model))