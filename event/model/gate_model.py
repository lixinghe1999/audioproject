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
from model.resnet34 import ResNet
# from vanilla_model import EncoderLayer
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
            # self.gate_audio = nn.Linear(self.bottle_neck*2, 12)
            # self.gate_image = nn.Linear(self.bottle_neck*2, 12)
        # Option2, another network: conv + max + linear
        else:
            self.gate_audio = ResNet(1, layers=(1, 1, 1, 1), num_classes=12)
            self.gate_image = ResNet(3, layers=(1, 1, 1, 1), num_classes=12)
            # self.gate_audio = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(4, 4)),
            #                 nn.MaxPool2d(kernel_size=(7, 7)), nn.Flatten(start_dim=1), nn.Linear(64, 12)])
            # self.gate_image = nn.Sequential(*[nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(4, 4)),
            #                 nn.MaxPool2d(kernel_size=(7, 7)), nn.Flatten(start_dim=1), nn.Linear(64, 12)])
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
            logits_image = logits_audio[:, 12:]
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
        def helper_1(modal1, modal2, b):
            for i in range(blocks):
                predict_label = self.projection(torch.cat([modal1[i][b], modal2[-1][b]], dim=-1))
                predict_label = torch.argmax(predict_label, dim=-1).item()
                if predict_label == label[b]:
                    for j in range(blocks):
                        predict_label = self.projection(torch.cat([modal1[i][b], modal2[-j-1][b]], dim=-1))
                        predict_label = torch.argmax(predict_label, dim=-1).item()
                        if predict_label != label[b]:
                            return i, blocks - j
            return blocks-1, blocks-1
        def helper_2(modal1, modal2, b):
            for i in range(blocks):
                predict_label = self.projection(torch.cat([modal2[-1][b], modal1[i][b]], dim=-1))
                predict_label = torch.argmax(predict_label, dim=-1).item()
                if predict_label == label[b]:
                    for j in range(blocks):
                        predict_label = self.projection(torch.cat([modal2[-j-1][b], modal1[i][b]], dim=-1))
                        predict_label = torch.argmax(predict_label, dim=-1).item()
                        if predict_label != label[b]:
                            return i, blocks - j
            return blocks-1, blocks-1

        batch = len(label)
        blocks = len(output_cache['audio'])
        gate_label = torch.zeros(batch, 2, blocks, dtype=torch.int8)
        for b in range(batch):
            i1, j1 = helper_1(output_cache['audio'], output_cache['image'], b)
            j2, i2 = helper_2(output_cache['image'], output_cache['audio'], b)
            if (i1 + j1) < (i2 + j2):
                gate_label[b, 0, i1] = 1
                gate_label[b, 1, j1] = 1
            else:
                gate_label[b, 0, i2] = 1
                gate_label[b, 1, j2] = 1
        return gate_label
    def gate_train(self, audio, image, label):
        output_cache, output = self.forward(audio, image, 'no_exit') # get all the possibilities
        gate_label = self.label(output_cache, label).to('cuda')
        gate_label = torch.argmax(gate_label, dim=-1)
        output, gate_a, gate_i = self.gate(audio, image, output_cache)
        loss_c1 = nn.functional.cross_entropy(gate_a, gate_label[:, 0]) # compression-level loss
        loss_c2 = nn.functional.cross_entropy(gate_i, gate_label[:, 1])  # compression-level loss
        print('gate acc:', (torch.argmax(gate_a, dim=-1) == gate_label[:, 0]).sum() / len(gate_label),
              (torch.argmax(gate_i, dim=-1) == gate_label[:, 1]).sum() / len(gate_label))
        output = self.projection(output)
        loss_r = nn.functional.cross_entropy(output, label) # recognition-level loss
        print('compress:', (torch.argmax(gate_a, dim=-1).float().mean() + 1)/12 ,
              (torch.argmax(gate_i, dim=-1).float().mean() + 1)/12)
        print('acc:', (torch.argmax(output, dim=-1) == label).sum() / len(label))
        print(loss_c1.item(), loss_c2.item(), loss_r.item())
        loss = loss_c1 * 0.5 + loss_c2 * 0.3
        loss.backward()
        return loss

    def acculmulative_loss(self, output_cache, label, criteria):
        loss = 0
        for i, embed in enumerate(output_cache['bottle_neck']):
            output = self.projection(torch.mean(embed, dim=1))
            loss += i/12 * criteria(output, label)
        return loss

    @autocast()
    def forward(self, x, y, mode='dynamic'):
        audio = x
        image = y
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

        # first block
        audio = self.audio.v.blocks[0](audio)
        audio_norm = self.audio.v.norm(audio)
        audio_norm = (audio_norm[:, 0] + audio_norm[:, 1]) / 2
        output_cache['audio'].append(audio_norm)

        image = self.image.v.blocks[0](image)
        image_norm = self.image.v.norm(image)
        image_norm = (image_norm[:, 0] + image_norm[:, 1]) / 2
        output_cache['image'].append(image_norm)

        if mode == 'dynamic':
            self.exit = torch.randint(12, (2, 1))
        elif mode == 'no_exit':
            # by default, no exit
            self.exit = torch.tensor([11, 11])
        elif mode == 'gate':
            # not implemented yet
            gate_a, gate_i = self.gate(x, y, output_cache)
            self.exit = torch.argmax(torch.cat([gate_a, gate_i], dim=0), dim=-1)

        # bottleneck_token = self.bottleneck_token.expand(B, -1, -1)
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.v.blocks[1:], self.image.v.blocks[1:])):
            if i < self.exit[0].item():
                audio = blk_a(audio)
                audio_norm = self.audio.v.norm(audio)
                audio_norm = (audio_norm[:, 0] + audio_norm[:, 1]) / 2
                output_cache['audio'].append(audio_norm)
            if i < self.exit[1].item():
                image = blk_i(image)
                image_norm = self.image.v.norm(image)
                image_norm = (image_norm[:, 0] + image_norm[:, 1]) / 2
                output_cache['image'].append(image_norm)
            # if i >= self.fusion_stage:
            #     bottleneck_token = self.bottleneck[i - self.fusion_stage](
            #         torch.cat((bottleneck_token, audio, image), dim=1))
            #     output_cache['bottle_neck'].append(bottleneck_token)
        # output = self.projection(torch.mean(bottleneck_token, dim=1))
        audio = output_cache['audio'][-1]
        image = output_cache['image'][-1]
        output = self.projection(torch.cat([audio, image], dim=-1))
        return output_cache, output
if __name__ == "__main__":
    num_cls = 100
    device = 'cpu'
    model = AVnet_Gate().to(device)
    model.eval()
    audio = torch.zeros(1, 384, 128).to(device)
    image = torch.zeros(1, 3, 224, 224).to(device)

    with torch.no_grad():
        for i in range(20):
            if i == 1:
                t_start = time.time()
            model(audio, image, 'no_exit')
    print((time.time() - t_start) / 19)