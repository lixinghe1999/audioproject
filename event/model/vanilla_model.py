'''
We implement multi-modal dynamic network here
'''
import torch.nn as nn
import torch
import torchaudio
from torch.cuda.amp import autocast
# from model.modified_resnet import ModifiedResNet
# from model.resnet34 import ResNet, BasicBlock
from model.ast_vit import ASTModel, VITModel
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
class BottleNeck_attention(nn.Module):
    def __init__(self, d_model, bottleneck):
        super(BottleNeck_attention, self).__init__()
        self.bottleneck = bottleneck
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    def forward(self, q, k, v):
        # dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 3. do scale dot product to compute similarity
        # out, attention = self.attention(q, k, v, mask=mask)
        batch_size, length, d_tensor = q.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)
        score = (q[:, :self.bottleneck, :] @ k_t) / d_tensor**0.5  # scaled dot product
        score = nn.functional.softmax(score)

        v = score @ v
        return v


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, bottleneck, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = BottleNeck_attention(d_model=d_model, bottleneck=bottleneck)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class AVnet(nn.Module):
    def __init__(self):
        super(AVnet, self).__init__()
        self.audio = ASTModel(input_tdim=384, audioset_pretrain=False, verbose=True, model_size='base224')
        self.image = VITModel(model_size='base224')

        self.original_embedding_dim = self.audio.v.pos_embed.shape[2]
        self.bottleneck_token = nn.Parameter(torch.zeros(1, 4, self.original_embedding_dim))
        self.fusion_stage = 6
        self.bottleneck = nn.ModuleList([EncoderLayer(self.original_embedding_dim, 2048, 4, 0.1) for _ in range(12-self.fusion_stage)])
        self.projection = nn.Sequential(nn.LayerNorm(self.original_embedding_dim * 2),
                                      nn.Linear(self.original_embedding_dim * 2, 309))
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

        cls_tokens = self.image.v.cls_token.expand(B, -1, -1)
        dist_token = self.image.v.dist_token.expand(B, -1, -1)
        image = torch.cat((cls_tokens, dist_token, image), dim=1)
        image = image + self.image.v.pos_embed
        image = self.image.v.pos_drop(image)

        bottleneck_token = self.bottleneck_token.expand(B, -1, -1)
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.v.blocks, self.image.v.blocks)):
            audio = blk_a(audio)
            image = blk_i(image)
            if i >= self.fusion_stage:
                bottleneck_token = self.bottleneck[i-self.fusion_stage](torch.cat((bottleneck_token, audio, image), dim=1))

        audio = self.audio.v.norm(audio)
        audio = (audio[:, 0] + audio[:, 1]) / 2
        image = self.image.v.norm(image)
        image = (image[:, 0] + image[:, 1]) / 2

        output = self.projection(torch.cat([audio, image], dim=-1))
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
