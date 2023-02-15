'''
We implement multi-modal dynamic network here
1. Early-exit: first concat two modal, (optional align, not need if the same Layer get same dimension) then add an early-exit branch
2. Flexible Early-exit: Each modal get early-exit for each layer, output = (A + V)/2.
 How to do backpropagation?
 1) all early-exit supervised by same ground-truth (seems decompose two modal?) almost the same as common early-exit
 2) pre-defined all potential combination (seems too many loss?)
 3) utilize modality fusion to get early-exit

'''
import torch.nn as nn
import torch
from model.modified_resnet import ModifiedResNet
from model.resnet34 import ResNet, BasicBlock
class AVnet(nn.Module):
    def __init__(self, exit=True, threshold=0.9, num_cls=309):
        '''
        :param exit: True - with exit, normally for testing, False - no exit, normally for training
        :param threshold: confidence to continue calculation, can be Integer or List
        :param num_cls: number of class
        '''
        super(AVnet, self).__init__()
        self.exit = exit
        self.threshold = threshold
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

        self.early_exit1 = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(64 * 2, num_cls)])
        self.early_exit2 = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(128 * 2, num_cls)])
        self.early_exit3 = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(256 * 2, num_cls)])
    def get_audio_params(self):
        parameters = [
            {'params': self.audio.parameters()},
            {'params': self.mmtm1.parameters()},
            {'params': self.mmtm2.parameters()},
            {'params': self.mmtm3.parameters()},
        ]
        return parameters
    def get_image_params(self):
        parameters = [
            {'params': self.image.parameters()},
            {'params': self.mmtm1.parameters()},
            {'params': self.mmtm2.parameters()},
            {'params': self.mmtm3.parameters()},
        ]
        return parameters
    def preprocessing_audio(self, audio):
        spec = torch.stft(audio.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=torch.hann_window(self.win_length, device=audio.device),
                          pad_mode='reflect', normalized=self.normalized, onesided=True, return_complex=True)
        spec = torch.abs(spec)
        spec = torch.log(spec + 1e-7)
        mean = torch.mean(spec)
        std = torch.std(spec)
        spec = (spec - mean) / (std + 1e-9)
        spec = torch.nn.functional.interpolate(spec.unsqueeze(1), size=self.spec_scale, mode='bilinear')
        return spec
    def forward(self, audio, image):
        output_cache = []
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
        image = self.image.layer1(image)
        early_output = self.early_exit1(torch.cat([audio, image], dim=1))
        output_cache.append(early_output)
        if self.exit and self.threshold > 0:
            print(early_output.shape)
            print(early_output)
            confidence = torch.softmax(early_output, dim=1).max()
            print(confidence)
            if confidence > self.threshold:
                return output_cache

        audio = self.audio.layer2(audio)
        image = self.image.layer2(image)
        early_output = self.early_exit2(torch.cat([audio, image], dim=1))
        output_cache.append(early_output)
        if self.exit and self.threshold > 0:
            confidence = torch.max(torch.softmax(early_output, dim=1), dim=1)
            if confidence > self.threshold:
                return output_cache

        audio = self.audio.layer3(audio)
        image = self.image.layer3(image)
        early_output = self.early_exit3(torch.cat([audio, image], dim=1))
        output_cache.append(early_output)
        if self.exit and self.threshold > 0:
            confidence = torch.max(torch.softmax(early_output, dim=1), dim=1)
            if confidence > self.threshold:
                return output_cache

        audio = self.audio.layer4(audio)
        image = self.image.layer4(image)

        audio = self.audio.avgpool(audio)
        image = self.audio.avgpool(image)
        audio = torch.flatten(audio, 1)
        image = torch.flatten(image, 1)
        output = (self.audio.fc(audio) + self.image.fc(image)) / 2
        output_cache.append(output)
        return output_cache
class AVnet_Flex(nn.Module):
    def __init__(self, exit=True, threshold=0.9, num_cls=309):
        '''
        :param exit: True - with exit, normally for testing, False - no exit, normally for training
        :param threshold: confidence to continue calculation, can be Integer or List
        :param num_cls: number of class
        '''
        super(AVnet_Flex, self).__init__()
        self.exit = exit
        self.threshold = threshold
        self.audio_exit = False
        self.image_exit = False

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

        self.early_exit1a = nn.Sequential(
            *[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(64, num_cls)])
        self.early_exit1b = nn.Sequential(
            *[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(64, num_cls)])

        self.early_exit2a = nn.Sequential(
            *[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(128, num_cls)])
        self.early_exit2b = nn.Sequential(
            *[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(128, num_cls)])

        self.early_exit3a = nn.Sequential(
            *[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(256, num_cls)])
        self.early_exit3b = nn.Sequential(
            *[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(256, num_cls)])

        self.early_exit4a = nn.Sequential(
            *[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(512, num_cls)])
        self.early_exit4b = nn.Sequential(
            *[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(512, num_cls)])
    def preprocessing_audio(self, audio):
        spec = torch.stft(audio.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=torch.hann_window(self.win_length, device=audio.device),
                          pad_mode='reflect', normalized=self.normalized, onesided=True, return_complex=True)
        spec = torch.abs(spec)
        spec = torch.log(spec + 1e-7)
        spec = (spec - torch.mean(spec)) / (torch.std(spec) + 1e-9)
        spec = torch.nn.functional.interpolate(spec.unsqueeze(1), size=self.spec_scale, mode='bilinear')
        return spec
    def inference_update(self, early_output, modal):
        if self.exit and self.threshold > 0:
            confidence = torch.softmax(early_output, dim=1)
            if torch.max(confidence, dim=1) > self.threshold:
                setattr(self, modal, True)
            else:
                setattr(self, modal, False)
    def forward(self, audio, image):
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
        self.inference_update(early_output, 'audio_exit')

        image = self.image.layer1(image)
        early_output = self.early_exit1b(image)
        output_cache['image'].append(early_output)
        self.inference_update(early_output, 'image_exit')

        if not self.audio_exit:
            audio = self.audio.layer2(audio)
            early_output = self.early_exit2a(audio)
            output_cache['audio'].append(early_output)
            self.inference_update(early_output, 'audio_exit')
        if not self.image_exit:
            image = self.image.layer2(image)
            early_output = self.early_exit2b(image)
            output_cache['image'].append(early_output)
            self.inference_update(early_output, 'image_exit')

        if not self.audio_exit:
            audio = self.audio.layer3(audio)
            early_output = self.early_exit3a(audio)
            output_cache['audio'].append(early_output)
            self.inference_update(early_output, 'audio_exit')
        if not self.image_exit:
            image = self.image.layer3(image)
            early_output = self.early_exit3b(image)
            output_cache['image'].append(early_output)
            self.inference_update(early_output, 'image_exit')

        if not self.audio_exit:
            audio = self.audio.layer4(audio)
            early_output = self.early_exit4a(audio)
            output_cache['audio'].append(early_output)
            self.inference_update(early_output, 'audio_exit')
        if not self.image_exit:
            image = self.image.layer3(image)
            early_output = self.early_exit4b(image)
            output_cache['image'].append(early_output)
            self.inference_update(early_output, 'image_exit')
        return output_cache
if __name__ == "__main__":
    num_cls = 100
    model = AVnet(num_cls=100)
    audio = torch.zeros(16, 1, 220500)
    image = torch.zeros(16, 3, 224, 224)
    outputs = model(audio, image)
    for output in outputs:
        print(output.shape)