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
from model.vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning
from model.resnet34 import ResNet
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
    def __init__(self, option=1):
        super(Gate, self).__init__()
        # Option1, use the embedding of first block
        self.option = option
        self.bottle_neck = 768
        if self.option == 1:
            self.gate = nn.Linear(self.bottle_neck * 2, 24)
            # self.gate_audio = nn.Linear(self.bottle_neck * 2, 12)
            # self.gate_image = nn.Linear(self.bottle_neck * 2, 12)
        # Option2, another network: conv + max + linear
        else:
            self.gate_audio = ResNet(1, layers=(1, 1, 1, 1), num_classes=12)
            self.gate_image = ResNet(3, layers=(1, 1, 1, 1), num_classes=12)
    def forward(self, audio, image, output_cache):
        '''
        :param audio, image: raw data
        :param output_cache: dict: ['audio', 'image'] list -> 12 (for example) * [batch, bottle_neck]
         Or [batch, raw_data_shape] -> [batch, 3, 224, 224]
        :return: Gumbel_softmax decision
        '''
        if self.option == 1:
            gate_input = torch.cat([output_cache['audio'][0], output_cache['image'][0]], dim=-1)
            logits = self.gate(gate_input)
            logits_audio = logits[:, :12]
            logits_image = logits[:, 12:]
            # logits_audio = self.gate_audio(gate_input)
            # logits_image = self.gate_image(gate_input)
            y_soft, ret_audio, index = gumbel_softmax(logits_audio)
            y_soft, ret_image, index = gumbel_softmax(logits_image)
        else:
            logits_audio = self.gate_audio(audio.unsqueeze(1))
            y_soft, ret_audio, index = gumbel_softmax(logits_audio)
            logits_image = self.gate_image(image)
            y_soft, ret_image, index = gumbel_softmax(logits_image)

        if len(output_cache['audio']) == 1:
            return ret_audio, ret_image
        else:
            audio = torch.cat(output_cache['audio'], dim=-1)
            audio = (audio.reshape(-1, 12, self.bottle_neck) * ret_audio.unsqueeze(2)).mean(dim=1)
            image = torch.cat(output_cache['image'], dim=-1)
            image = (image.reshape(-1, 12, self.bottle_neck) * ret_image.unsqueeze(2)).mean(dim=1)
            return torch.cat([audio, image], dim=-1), ret_audio, ret_image
class AVnet_Gate(nn.Module):
    def __init__(self, gate_network=Gate(option=1), scale='base', pretrained=True):
        '''
        :param exit: True - with exit, normally for testing, False - no exit, normally for training
        :param gate_network: extra gate network
        :param num_cls: number of class
        '''
        super(AVnet_Gate, self).__init__()
        self.gate = gate_network
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
        self.projection = nn.Sequential(nn.LayerNorm(embed_dim*2),
                                        nn.Linear(embed_dim*2, 309))
    def get_parameters(self):
        parameter = [{'params': self.gate.parameters()},
                     {'params': self.projection.parameters()}]
        return parameter

    def label(self, output_cache, label, mode='flexible'):
        # 1. model -> switch between two modality -> predict [2, 1]
        # 2. main -> predict one branch [1, 12]
        # 3. flexible -> predict [2, 12] currently very close to model
        def helper_1(modal1, modal2, b):
            for i in range(blocks):
                predict_label = self.projection(torch.cat([modal1[i][b], modal2[-1][b]], dim=-1))
                predict_label = torch.argmax(predict_label, dim=-1).item()
                if predict_label == label[b]:
                    return i, blocks - 1
            return blocks-1, blocks-1
        def helper_2(modal1, modal2, b):
            for i in range(blocks):
                predict_label = self.projection(torch.cat([modal2[-1][b], modal1[i][b]], dim=-1))
                predict_label = torch.argmax(predict_label, dim=-1).item()
                if predict_label == label[b]:
                    return i, blocks - 1
            return blocks-1, blocks-1

        batch = len(label)
        blocks = len(output_cache['audio'])
        gate_label = torch.zeros(batch, 2, blocks, dtype=torch.int8)
        for b in range(batch):
            i1, j1 = helper_1(output_cache['audio'], output_cache['image'], b)
            j2, i2 = helper_2(output_cache['image'], output_cache['audio'], b)
            if mode == 'main':
                i, j = i1, blocks-1
            elif mode == 'model':
                if i1 < j2:
                    i = 0; j = blocks-1
                else:
                    i = blocks-1; j = 0
            else: # flexible
                if (i1 + j1) < (i2 + j2):
                    i = i1; j = j1
                elif (i1 + j1) > (i2 + j2):
                    i = i2; j = j2
                else:
                    if torch.rand(1)<0.5:
                        i = i1; j = j1
                    else:
                        i = i2; j = j2
            gate_label[b, 0, i] = 1; gate_label[b, 1, j] = 1
        return gate_label
    def gate_train(self, audio, image, label):
        '''
        We get three loss: computation loss, Gate label loss, Recognition loss
        '''
        output_cache, output = self.forward(audio, image, 'no_exit') # get all the possibilities
        gate_label = self.label(output_cache, label, mode='model').to('cuda')
        gate_label = torch.argmax(gate_label, dim=-1)
        output, gate_a, gate_i = self.gate(audio, image, output_cache)

        computation_penalty = torch.range(1, 12).to('cuda')
        loss_c = (gate_a * computation_penalty + gate_i * computation_penalty).mean()
        loss_g1 = nn.functional.cross_entropy(gate_a, gate_label[:, 0])
        loss_g2 = nn.functional.cross_entropy(gate_i, gate_label[:, 1])

        output = self.projection(output)
        loss_r = nn.functional.cross_entropy(output, label) # recognition-level loss

        compress = [(torch.argmax(gate_a, dim=-1).float().mean() + 1).item()/12 ,
              (torch.argmax(gate_i, dim=-1).float().mean() + 1).item()/12]
        acc = (torch.argmax(output, dim=-1) == label).sum().item() / len(label)

        # print(loss_c.item(), (loss_g1 + loss_g2).item(), loss_r.item())
        loss = loss_c * 0.3 + loss_g1 * 0.3 + loss_g2 * 0.3 + loss_r * 0.1
        # loss = loss_c * 0.5 + loss_g * 0.4 + loss_r * 0.2
        loss.backward()
        return [compress, acc]

    @autocast()
    def forward(self, audio, image, mode='dynamic'):
        output_cache = {'audio': [], 'image': []}

        B, audio = self.audio.preprocess(audio.unsqueeze(1))
        B, image = self.image.preprocess(image)

        # first block
        audio = self.audio.blocks[0](audio)
        audio_norm = self.audio.norm(audio)[:, 0]
        output_cache['audio'].append(audio_norm)

        image = self.image.blocks[0](image)
        image_norm = self.image.norm(image)[:, 0]
        output_cache['image'].append(image_norm)

        if mode == 'dynamic':
            self.exit = torch.randint(12, (2, 1))
        elif mode == 'no_exit':
            # by default, no exit
            self.exit = torch.tensor([11, 11])
        elif mode == 'gate':
            # not implemented yet
            gate_a, gate_i = self.gate(audio_norm, image_norm, output_cache)
            self.exit = torch.argmax(torch.cat([gate_a, gate_i], dim=0), dim=-1)
        else: # directly get the exit
            self.exit = mode

        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks[1:], self.image.blocks[1:])):
            if i < self.exit[0].item():
                audio = blk_a(audio)
                audio_norm = self.audio.norm(audio)[:, 0]
                output_cache['audio'].append(audio_norm)
            if i < self.exit[1].item():
                image = blk_i(image)
                image_norm = self.image.norm(image)[:, 0]
                output_cache['image'].append(image_norm)

        audio = output_cache['audio'][-1]
        image = output_cache['image'][-1]
        output = self.projection(torch.cat([audio, image], dim=-1))
        return output_cache, output