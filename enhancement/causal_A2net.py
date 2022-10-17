import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
from torch.nn.modules.utils import _pair
class IMU_branch(nn.Module):
    def __init__(self, inference=False):
        super(IMU_branch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(4, 1), dilation=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(4, 1), dilation=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1)),
            nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(4, 1), dilation=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.inference = inference
        if not inference:
            self.conv5 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            self.conv6 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True))
            self.conv7 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=(3, 2), stride=(3, 2)),
                nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True))


    def forward(self, x):
        # down-sample
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_mid = self.conv4(x)
        # up-sample with supervision
        if not self.inference:
            x = self.conv5(x_mid)
            x = self.conv6(x)
            x = self.conv7(x)
            x = F.pad(x, [0, 1, 0, 0])
            return x_mid, x
        else:
            return x_mid
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

class Encoder(nn.Module):
    def __init__(self, filters=[16, 32, 64, 128],
                 kernels=[3, 3, 3, 3], max_pooling={2:[1,3], 3:[2,1]}):
        super(Encoder, self).__init__()
        self.num_layers = len(filters)
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_channel = 1
            else:
                input_channel = filters[i-1]
            output_channel = filters[i]
            kernel = kernels[i]
            conv = CausalConv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel)
            layer = nn.Sequential(
            conv, nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True))
            layers.append(layer)
            if i in max_pooling:
                layers.append(nn.MaxPool2d(kernel_size=max_pooling[i]))
            self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
class Decoder(nn.Module):
    def __init__(self, input_channel, filters=[16, 32, 64, 128],
                 kernels=[3, 3, 3, 3], max_pooling={2:[1,3], 3:[2,1]}):
        super(Decoder, self).__init__()
        self.num_layers = len(filters)
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_channel = input_channel
            else:
                input_channel = filters[i-1]
            output_channel = filters[i]
            kernel = kernels[i]
            conv = CausalConv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel)
            layer = nn.Sequential(
                conv, nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True))
            layers.append(layer)
            if i in max_pooling:
                layers.append(nn.ConvTranspose2d(output_channel, output_channel, kernel_size=max_pooling[i], stride=max_pooling[i]))
        layers.append(nn.Conv2d(filters[-1], 1, kernel_size=1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
class Causal_A2net(nn.Module):
    def __init__(self):
        super(Causal_A2net, self).__init__()
        self.model_acc = Encoder(filters=[16, 32, 64], kernels=[3, 3, 3],
                            max_pooling={2: [3, 1]})
        self.model_audio = Encoder(filters=[16, 32, 64, 128],
                              kernels=[5, 5, 5, 3],
                              max_pooling={0: [2, 1], 1: [2, 1], 2: [2, 1], 3: [3, 1]})
        self.model_fusion = Decoder(input_channel=128 + 64, filters=[128, 64, 32, 16],
                               kernels=[5, 5, 5, 3],
                               max_pooling={0: [3, 1], 1: [2, 1], 2: [2, 1], 3: [2, 1]})
    def forward(self, audio, acc):
        mixture = torch.cat([self.model_audio(audio), self.model_acc(acc)], dim=1)
        clean_audio = self.model_fusion(mixture)
        return clean_audio
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
    audio = torch.rand(1, 1, 264, 151)
    acc = torch.rand(1, 1, 33, 151)
    model = Causal_A2net()
    clean_audio = model(audio, acc)
    print(clean_audio.shape)
    ckpt = model.state_dict()
    torch.save(ckpt, 'causal.pth')
    size_all_mb = model_size(model)
    print('model size: {:.3f}MB'.format(size_all_mb))
    latency = model_speed(model, [audio, acc])
    print('model latency: {:.3f}S'.format(latency))
