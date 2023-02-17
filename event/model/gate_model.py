'''
We implement multi-modal dynamic network here
1. Early-exit: first concat two modal, (optional align, not need if the same Layer get same dimension) then add an early-exit branch
2. Flexible Early-exit: Each modal get early-exit for each layer, output = (A + V)/2.
 How to do backpropagation?
 1) all early-exit supervised by same ground-truth (seems decompose two modal?) almost the same as common early-exit
 2) pre-defined all potential combination (seems too many loss?)
 3) utilize modality fusion to get early-exit

'''
import time
import torch.nn as nn
import torch
from model.modified_resnet import ModifiedResNet
from model.resnet34 import ResNet, BasicBlock
class AVnet_Gate(nn.Module):
    def __init__(self, exit=False, num_cls=309):
        '''
        :param exit: True - with exit, normally for testing, False - no exit, normally for training
        :param threshold: confidence to continue calculation, can be Integer or List
        :param num_cls: number of class
        '''
        super(AVnet_Gate, self).__init__()
        self.exit = exit
        self.audio_exit = False
        self.image_exit = False
        self.bottle_neck = 128

        self.audio = ResNet(img_channels=1, layers=(3, 4, 6, 3), block=BasicBlock, num_classes=num_cls)
        self.n_fft = 512
        self.hop_length = 512
        self.win_length = 512
        self.spec_scale = 224
        self.normalized = True
        self.onesided = True

        self.image = ResNet(img_channels=3, layers=(3, 4, 6, 3), block=BasicBlock, num_classes=1000)
        self.image.load_state_dict(torch.load('resnet34.pth'))
        self.image.fc = torch.nn.Linear(512, num_cls)

        self.early_exit1a = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(64, self.bottle_neck)])
        self.early_exit1b = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(64, self.bottle_neck)])

        self.early_exit2a = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(128, self.bottle_neck)])
        self.early_exit2b = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(128, self.bottle_neck)])

        self.early_exit3a = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(256, self.bottle_neck)])
        self.early_exit3b = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(256, self.bottle_neck)])

        self.early_exit4a = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(512, self.bottle_neck)])
        self.early_exit4b = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(512, self.bottle_neck)])

        self.projection = nn.Linear(self.bottle_neck * 2, num_cls)
        self.gate = nn.Linear(self.bottle_neck * 2, 2)
    def preprocessing_audio(self, audio):
        spec = torch.stft(audio.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=torch.hann_window(self.win_length, device=audio.device),
                          pad_mode='reflect', normalized=self.normalized, onesided=True, return_complex=True)
        spec = torch.abs(spec)
        spec = torch.log(spec + 1e-7)
        spec = torch.nn.functional.interpolate(spec.unsqueeze(1), size=self.spec_scale, mode='bilinear')
        return spec
    def inference_update(self, early_output, modal, random_thres=1.0):
        if self.exit:
            # exit based on threshold
            work_load = self.gate(early_output)
            attributes = ['audio_exit', 'image_exit']
            for conf, attr in zip(work_load, attributes):
                if conf > self.threshold:
                    setattr(self, modal, True)
                else:
                    setattr(self, modal, False)
        else:
            random_number = torch.rand(1)
            if random_number.item() < random_thres:
                setattr(self, modal, True)
    def forward(self, audio, image):
        self.audio_exit = False
        self.image_exit = False
        output_cache = {'audio': [], 'image': []}
        audio = self.preprocessing_audio(audio)
        audio = self.audio.conv1(audio)
        audio = self.audio.bn1(audio)
        audio = self.audio.relu(audio)
        audio = self.audio.maxpool(audio)

        image = self.image.conv1(image)
        image = self.image.bn1(image)
        image = self.image.relu(image)
        image = self.image.maxpool(image)

        audio = self.audio.layer1(audio)
        early_output = self.early_exit1a(audio)
        output_cache['audio'].append(early_output)
        image = self.image.layer1(image)
        early_output = self.early_exit1b(image)
        output_cache['image'].append(early_output)
        self.inference_update(torch.cat([output_cache['audio'][-1], output_cache['image'][-1]], dim=1),
                              'audio_exit', 0.25)
        self.inference_update(torch.cat([output_cache['audio'][-1], output_cache['image'][-1]], dim=1),
                              'image_exit', 0.25)
        print(len(output_cache['audio']) + len(output_cache['image']))
        if not self.audio_exit:
            audio = self.audio.layer2(audio)
            early_output = self.early_exit2a(audio)
            output_cache['audio'].append(early_output)
            self.inference_update(torch.cat([output_cache['audio'][-1], output_cache['image'][-1]], dim=1),
                                'audio_exit', 0.33)
        if not self.image_exit:
            image = self.image.layer2(image)
            early_output = self.early_exit2b(image)
            output_cache['image'].append(early_output)
            self.inference_update(torch.cat([output_cache['audio'][-1], output_cache['image'][-1]], dim=1),
                                  'image_exit', 0.33)
        print(len(output_cache['audio']) + len(output_cache['image']))

        if not self.audio_exit:
            audio = self.audio.layer3(audio)
            early_output = self.early_exit3a(audio)
            output_cache['audio'].append(early_output)
            self.inference_update(torch.cat([output_cache['audio'][-1], output_cache['image'][-1]], dim=1),
                                  'audio_exit', 0.5)
        if not self.image_exit:
            image = self.image.layer3(image)
            early_output = self.early_exit3b(image)
            output_cache['image'].append(early_output)
            self.inference_update(torch.cat([output_cache['audio'][-1], output_cache['image'][-1]], dim=1),
                                  'image_exit', 0.5)
        print(len(output_cache['audio']) + len(output_cache['image']))

        if not self.audio_exit:
            audio = self.audio.layer4(audio)
            early_output = self.early_exit4a(audio)
            output_cache['audio'].append(early_output)
        if not self.image_exit:
            image = self.image.layer4(image)
            early_output = self.early_exit4b(image)
            output_cache['image'].append(early_output)
        output = self.projection(torch.cat([output_cache['audio'][-1], output_cache['image'][-1]], dim=1))
        print(len(output_cache['audio']) + len(output_cache['image']))
        return output_cache, output
if __name__ == "__main__":
    num_cls = 100
    device = 'cuda'
    model = AVnet_Gate(num_cls=100).to(device)
    model.eval()
    audio = torch.zeros(1, 1, 160000).to(device)
    image = torch.zeros(1, 3, 224, 224).to(device)

    with torch.no_grad():
        for i in range(20):
            if i == 1:
                t_start = time.time()
            model(audio, image)
    print((time.time() - t_start) / 19)