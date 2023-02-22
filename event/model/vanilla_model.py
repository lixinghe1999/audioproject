'''
We implement multi-modal dynamic network here
'''
import torch.nn as nn
import torch
import torchaudio
from torch.cuda.amp import autocast
from model.modified_resnet import ModifiedResNet
from model.resnet34 import ResNet, BasicBlock
from model.ast_vit import ASTModel, VITModel
class AVnet(nn.Module):
    def __init__(self):
        super(AVnet, self).__init__()
        self.audio = ASTModel(input_tdim=384, audioset_pretrain=False, verbose=True, model_size='base224')
        self.image = VITModel(model_size='base224')
        # self.audio = ResNet(img_channels=1, layers=(3, 4, 6, 3), block=BasicBlock, num_classes=num_cls)
        # self.image = ModifiedResNet()
        # self.image.load_state_dict(torch.load('resnet50.pth'))
        # self.image.fc = torch.nn.Linear(1024, num_cls)

        # self.mmtm1 = MMTM(128, 512, 4)
        # self.mmtm2 = MMTM(256, 1024, 4)
        # self.mmtm3 = MMTM(512, 2048, 4)
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

    @autocast()
    def forward(self, audio, image):
        B = audio.shape[0]
        audio = audio.unsqueeze(1)
        audio = audio.transpose(2, 3)

        audio = self.audio.v.patch_embed(audio)
        image = self.image.v.patch_embed(image)

        cls_tokens = self.audio.v.cls_token.expand(B, -1, -1)
        dist_token = self.audio.v.dist_token.expand(B, -1, -1)
        audio = torch.cat((cls_tokens, dist_token, audio), dim=1)
        audio = audio + self.audio.v.pos_embed
        audio = self.audio.v.pos_drop(audio)
        for blk in self.audio.v.blocks:
            audio = blk(audio)
        audio = self.audio.v.norm(audio)
        audio = (audio[:, 0] + audio[:, 1]) / 2
        audio = self.audio.mlp_head(audio)

        cls_tokens = self.image.v.cls_token.expand(B, -1, -1)
        dist_token = self.image.v.dist_token.expand(B, -1, -1)
        image = torch.cat((cls_tokens, dist_token, image), dim=1)
        image = image + self.image.v.pos_embed
        image = self.image.v.pos_drop(image)
        for blk in self.image.v.blocks:
            image = blk(image)
        image = self.image.v.norm(image)
        image = (image[:, 0] + image[:, 1]) / 2
        image = self.image.mlp_head(image)
        output = (audio + image) / 2
        return output
class SingleNet(nn.Module):
    def __init__(self, modality='A', num_cls=309):
        '''
        :param modality: A or V
        :param layers: resnet18: [2, 2, 2, 2] resnet34: [3, 4, 6, 3]
        :param num_cls: number of class
        '''
        super(SingleNet, self).__init__()
        self.modality = modality
        if modality == 'A':
            self.net = ResNet(img_channels=1, layers=(3, 4, 6, 3), block=BasicBlock, num_classes=num_cls)
            self.n_fft = 512
            self.hop_length = 512
            self.win_length = 512
            self.spec_scale = 224
            self.normalized = True
            self.onesided = True
        else:
            self.net = ResNet(img_channels=3, layers=(3, 4, 6, 3), block=BasicBlock, num_classes=1000)
            self.net.load_state_dict(torch.load('resnet34.pth'))
            self.net.fc = torch.nn.Linear(512, num_cls)
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
    def forward(self, data):
        if self.modality == 'A':
            data = self.preprocessing_audio(data)

        data = self.net.conv1(data)
        data = self.net.bn1(data)
        data = self.net.relu(data)
        data = self.net.maxpool(data)

        data = self.net.layer1(data)
        data = self.net.layer2(data)
        data = self.net.layer3(data)
        data = self.net.layer4(data)

        data = self.net.avgpool(data)
        data = torch.flatten(data, 1)
        data = self.net.fc(data)
        return data
if __name__ == "__main__":
    num_cls = 100
    model = AVnet()
    audio = torch.zeros(16, 1, 220500)
    image = torch.zeros(16, 3, 224, 224)
    outputs = model(audio, image)
    print(outputs.shape)
