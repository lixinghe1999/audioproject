'''
Baseline1:
add early-exit on each block
Two modality will always have same computation
'''
import time
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from model.vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning
class AVnet_Exit(nn.Module):
    def __init__(self, scale='base', pretrained=True):
        super(AVnet_Exit, self).__init__()
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
        self.projection = nn.ModuleList([nn.Sequential(nn.LayerNorm(embed_dim*2), nn.Linear(embed_dim*2, 309))
                                         for _ in range(12)])
    def get_parameters(self):
        parameter = [{'params': self.projection.parameters()}]
        return parameter

    @autocast()
    def forward(self, audio, image):
        B, audio = self.audio.preprocess(audio.unsqueeze(1))
        B, image = self.image.preprocess(image)
        output = []
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            audio = blk_a(audio)
            image = blk_i(image)
            audio_norm = self.audio.norm(audio)
            image_norm = self.image.norm(image)
            x = torch.cat([audio_norm[:, 0], image_norm[:, 0]], dim=1)
            x = torch.flatten(x, start_dim=1)
            output.append(self.projection[i](x))
        return output

