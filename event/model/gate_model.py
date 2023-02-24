'''
We implement multi-modal dynamic network here
We get three modes
1. Training main: randomly select exit (or bootstrapped?), without using gate network, exit = False
2. Training gate: without exit, only train the gate network, exit = False
3. Inference: use trained gate network to do inference, exit = True
'''
import time
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from model.ast_vit import ASTModel, VITModel
from model.vanilla_model import EncoderLayer
def gumbel_softmax(logits, tau=1, hard=False, dim=1, training=True):
    """ See `torch.nn.functional.gumbel_softmax()` """
    # if training:
    # gumbels = -torch.empty_like(logits,
    #                             memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    # gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # # else:
    # #     gumbels = logits
    # y_soft = gumbels.softmax(dim)

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    with torch.no_grad():
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        #  **test**
        # index = 0
        # y_hard = torch.Tensor([1, 0, 0, 0]).repeat(logits.shape[0], 1).cuda()
    ret = y_hard - y_soft.detach() + y_soft
    return y_soft, ret, index
class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()
        self.bottle_neck = 128
        self.gate_audio = nn.Linear(self.bottle_neck * 2, 4)
        self.gate_image = nn.Linear(self.bottle_neck * 2, 4)
    def forward(self, output_cache):
        '''
        :param output_cache: dict: ['audio', 'image'] list -> 4 (for example) * [batch, bottle_neck]
        :return: Gumbel_softmax decision
        '''

        gate_input = torch.cat([output_cache['audio'][0], output_cache['image'][0]], dim=-1)

        logits_audio = self.gate_audio(gate_input)
        y_soft, ret_audio, index = gumbel_softmax(logits_audio)
        audio = torch.cat(output_cache['audio']) * ret_audio

        logits_image = self.gate_image(gate_input)
        y_soft, ret_image, index = gumbel_softmax(logits_image)
        image = torch.cat(output_cache['image']) * ret_image
        return torch.cat([audio, image]), torch.cat([ret_audio, ret_image])
class AVnet_Gate(nn.Module):
    def __init__(self, gate_network=None):
        '''
        :param exit: True - with exit, normally for testing, False - no exit, normally for training
        :param gate_network: extra gate network
        :param num_cls: number of class
        '''
        super(AVnet_Gate, self).__init__()
        self.gate = gate_network

        self.audio = ASTModel(input_tdim=384, audioset_pretrain=False, verbose=True, model_size='base224')
        self.image = VITModel(model_size='base224')

        self.original_embedding_dim = self.audio.v.pos_embed.shape[2]
        # self.bottleneck_token = nn.Parameter(torch.zeros(1, 4, self.original_embedding_dim))
        # self.fusion_stage = 6
        # self.bottleneck = nn.ModuleList(
        #     [EncoderLayer(self.original_embedding_dim, 512, 4, 0.1) for _ in range(12 - self.fusion_stage)])
        self.projection = nn.Sequential(nn.LayerNorm(self.original_embedding_dim*2),
                                        nn.Linear(self.original_embedding_dim*2, 309))
    def fusion_parameter(self):
        parameter = [# {'params': self.bottleneck_token},
                     #{'params': self.bottleneck.parameters()},
                     {'params': self.projection.parameters()}]
        return parameter

    def label(self, output_cache, label):
        # label rule -> according the task
        # classification:
        # 1. starting from no compression: i & j
        # 2. if correct, randomly compress one modality
        # 3. if not, output the previous compression level
        i, j = len(output_cache['audio']) - 1, len(output_cache['image']) - 1
        self.global_min = 1000
        self.global_i, self.global_j = i, j
        def helper(i, j):
            if i < 0 or j < 0:
                pass
            else:
                audio = self.audio.v.norm(output_cache['audio'][i])
                audio = (audio[:, 0] + audio[:, 1]) / 2
                image = self.image.v.norm(output_cache['image'][j])
                image = (image[:, 0] + image[:, 1]) / 2
                predict_label = self.projection(torch.cat([audio, image], dim=-1))
                if torch.argmax(predict_label, dim=-1).cpu() == label:
                    print(i, j)
                    helper(i-1, j)
                    helper(i, j-1)
                    if (i+j) < self.global_min:
                        self.global_min = i + j
                        self.global_i = i
                        self.global_j = j
        helper(i, j)
        gate_label = torch.zeros(2, 4, dtype=torch.int8)
        gate_label[0, self.global_i] = 1
        gate_label[1, self.global_j] = 1
        return gate_label
    def gate_train(self, audio, image, label):
        output_cache, output = self.forward(audio, image) # get all the possibilities
        gate_label = self.label(output_cache, label) # [batch, 4 * 2]
        output, gate = self.gate(output_cache) # [batch, 4 * 2]
        loss_c = nn.functional.cross_entropy(gate, gate_label) # compression-level loss
        output = self.projection(output, dim=1)
        loss_r = nn.functional.cross_entropy(output, label) # recognition-level loss
        return loss_c, loss_r

    def acculmulative_loss(self, output_cache, label, criteria):
        loss = 0
        for i, embed in enumerate(output_cache['bottle_neck']):
            output = self.projection(torch.mean(embed, dim=1))
            loss += i/12 * criteria(output, label)
        return loss

    @autocast()
    def forward(self, audio, image, mode='dynamic'):
        if mode == 'dynamic':
            self.exit = torch.randint(12, (2, 1))
        elif mode == 'no_exit':
            # by default, no exit
            self.exit = torch.tensor([11, 11])
        elif mode == 'gate':
            # not implemented yet
            pass
        output_cache = {'audio': [], 'image': [], 'bottle_neck': []}

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

        # bottleneck_token = self.bottleneck_token.expand(B, -1, -1)
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.v.blocks, self.image.v.blocks)):
            if i <= self.exit[0].item():
                audio = blk_a(audio)
                output_cache['audio'].append(audio)
            if i <= self.exit[1].item():
                image = blk_i(image)
                output_cache['image'].append(image)
            # if i >= self.fusion_stage:
            #     bottleneck_token = self.bottleneck[i - self.fusion_stage](
            #         torch.cat((bottleneck_token, audio, image), dim=1))
            #     output_cache['bottle_neck'].append(bottleneck_token)
        # output = self.projection(torch.mean(bottleneck_token, dim=1))
        audio = self.audio.v.norm(audio)
        audio = (audio[:, 0] + audio[:, 1]) / 2
        image = self.image.v.norm(image)
        image = (image[:, 0] + image[:, 1]) / 2
        output = self.projection(torch.cat([audio, image], dim=-1))
        return output_cache, output
if __name__ == "__main__":
    num_cls = 100
    device = 'cuda'
    model = AVnet_Gate().to(device)
    model.eval()
    audio = torch.zeros(1, 1, 160000).to(device)
    image = torch.zeros(1, 3, 224, 224).to(device)

    with torch.no_grad():
        for i in range(20):
            if i == 1:
                t_start = time.time()
            model(audio, image)
    print((time.time() - t_start) / 19)