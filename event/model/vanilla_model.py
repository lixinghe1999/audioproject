'''
Baseline0:
'''
import time

import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from model.vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning
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
        k_t = k.transpose(1, 2)
        score = (q[:, :self.bottleneck, :] @ k_t) / d_tensor**0.5  # scaled dot product
        score = nn.functional.softmax(score)
        v = score @ v
        return v
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, bottleneck, drop_prob):
        super(EncoderLayer, self).__init__()
        self.bottleneck = bottleneck
        self.attention = BottleNeck_attention(d_model=d_model, bottleneck=bottleneck)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x[:, :self.bottleneck, :]
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
class Cross_attention(nn.Module):
    def __init__(self, d_model):
        super(Cross_attention, self).__init__()
        self.d_tensor = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        # dot product with weight matrices
        _x1, _x2 = x1, x2
        t1 = x1.shape[1]
        x = torch.cat([x1, x2], dim=1)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        k_t = k.transpose(1, 2)
        score = (q @ k_t) / self.d_tensor ** 0.5  # scaled dot product
        score = nn.functional.softmax(score)
        v = score @ v
        x1 = self.norm(v[:, :t1, :] + _x1)
        x2 = self.norm(v[:, t1:, :] + _x2)
        return x1, x2
class AVnet(nn.Module):
    def __init__(self, scale='base', pretrained=False):
        super(AVnet, self).__init__()
        if scale == 'base':
            config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                          pruning_loc=())
            embed_dim = 768
        else:
            config = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                          pruning_loc=())
            embed_dim = 384
        self.audio = AudioTransformerDiffPruning(config, imagenet_pretrain=pretrained)
        self.image = VisionTransformerDiffPruning(**config)
        if pretrained:
            self.image.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'], strict=False)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim * 2),
                                      nn.Linear(embed_dim * 2, 309))
        self.fusion_layer = 6
        self.fusion = nn.ModuleList([Cross_attention(embed_dim) for _ in range(12 - self.fusion_layer)])

    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()},
                     {'params': self.fusion.parameters()}]
        return parameter

    @autocast()
    def forward(self, audio, image):
        B, audio = self.audio.preprocess(audio.unsqueeze(1))
        B, image = self.image.preprocess(image)

        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            audio = blk_a(audio)
            image = blk_i(image)
            if i >= self.fusion_layer:
                audio, image = self.fusion[i-self.fusion_layer](audio, image)

        audio = self.audio.norm(audio)
        image = self.image.norm(image)
        x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x
if __name__ == "__main__":
    device = 'cuda'
    base_rate = 0.5
    pruning_loc = [3, 6, 9]
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]

    model = AVnet().to(device)

    model.eval()

    audio = torch.zeros(1, 384, 128).to(device)
    image = torch.zeros(1, 3, 224, 224).to(device)
    num_iterations = 100
    t_start = time.time()
    for _ in range(num_iterations):
        model(audio, image)
    print((time.time() - t_start)/num_iterations)
