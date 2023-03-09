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
from torch.cuda.amp import autocast
from model.vit import ASTModel, VITModel
from model.resnet34 import ResNet
class AVnet_Exit(nn.Module):
    def __init__(self):
        '''
        :param exit: True - with exit, normally for testing, False - no exit, normally for training
        :param gate_network: extra gate network
        :param num_cls: number of class
        '''
        super(AVnet_Exit, self).__init__()

        self.audio = ASTModel(input_tdim=384, audioset_pretrain=False, verbose=True, model_size='base224')
        self.image = VITModel(model_size='base224')

        self.original_embedding_dim = self.audio.v.pos_embed.shape[2]
        self.projection = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.original_embedding_dim*2),
                                        nn.Linear(self.original_embedding_dim*2, 309)) for _ in range(12)])
    def get_parameters(self):
        parameter = [{'params': self.projection.parameters()}]
        return parameter

    @autocast()
    def forward(self, x, y):
        audio = x
        image = y
        output_cache = {'audio': [], 'image': []}
        output = []

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

        # first block
        audio = self.audio.v.blocks[0](audio)
        audio_norm = self.audio.v.norm(audio)
        audio_norm = (audio_norm[:, 0] + audio_norm[:, 1]) / 2
        output_cache['audio'].append(audio_norm)

        image = self.image.v.blocks[0](image)
        image_norm = self.image.v.norm(image)
        image_norm = (image_norm[:, 0] + image_norm[:, 1]) / 2
        output_cache['image'].append(image_norm)

        # bottleneck_token = self.bottleneck_token.expand(B, -1, -1)
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.v.blocks, self.image.v.blocks)):
            audio = blk_a(audio)
            audio_norm = self.audio.v.norm(audio)
            audio_norm = (audio_norm[:, 0] + audio_norm[:, 1]) / 2
            output_cache['audio'].append(audio_norm)
            image = blk_i(image)
            image_norm = self.image.v.norm(image)
            image_norm = (image_norm[:, 0] + image_norm[:, 1]) / 2
            output_cache['image'].append(image_norm)

            output = self.projection[i](torch.cat([audio_norm, image_norm], dim=-1))
        return output_cache, output
if __name__ == "__main__":
    num_cls = 100
    device = 'cuda'
    model = AVnet_Flex(num_cls=100).to(device)
    model.eval()
    audio = torch.zeros(1, 1, 160000).to(device)
    image = torch.zeros(1, 3, 224, 224).to(device)

    with torch.no_grad():
        for i in range(20):
            if i == 1:
                t_start = time.time()
            model(audio, image)
    print((time.time() - t_start) / 19)