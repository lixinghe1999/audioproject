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
from model.modified_resnet import ModifiedResNet
from model.resnet34 import ResNet, BasicBlock
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
    def __init__(self, exit=False, gate_network=None, num_cls=309):
        '''
        :param exit: True - with exit, normally for testing, False - no exit, normally for training
        :param gate_network: extra gate network
        :param num_cls: number of class
        '''
        super(AVnet_Gate, self).__init__()
        self.exit = exit
        self.gate = gate_network
        if self.gate is None:
            self.exit = False
        self.audio_exit = False
        self.image_exit = False
        self.bottle_neck = 128

        self.audio = ResNet(img_channels=1, layers=(3, 4, 6, 3), block=BasicBlock, num_classes=num_cls)
        self.n_fft = 512
        self.hop_length = 512
        self.win_length = 512
        self.spec_scale = 224

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

    def preprocessing_audio(self, audio):
        spec = torch.stft(audio.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=torch.hann_window(self.win_length, device=audio.device),
                          pad_mode='reflect', normalized=True, onesided=True, return_complex=True)
        spec = torch.abs(spec)
        spec = torch.log(spec + 1e-7)
        spec = torch.nn.functional.interpolate(spec.unsqueeze(1), size=self.spec_scale, mode='bilinear')
        return spec
    def stem(self, audio, image):
        audio = self.preprocessing_audio(audio)
        audio = self.audio.conv1(audio)
        audio = self.audio.bn1(audio)
        audio = self.audio.relu(audio)
        audio = self.audio.maxpool(audio)

        image = self.image.conv1(image)
        image = self.image.bn1(image)
        image = self.image.relu(image)
        image = self.image.maxpool(image)
        return audio, image
    def inference_update(self, modal, gate=None, level=1, random_thres=1.0):
        if self.exit:
            # exit based on gate network
            compress_level = torch.argmax(gate)
            if level > compress_level:
                setattr(self, modal, True)
            else:
                setattr(self, modal, False)
        else:
            # randomly exit to train
            random_number = torch.rand(1)
            if random_number.item() < random_thres:
                setattr(self, modal, True)
    def label(self, output_cache, label):
        # label rule -> according the task
        # classification:
        # 1. starting from no compression: i & j
        # 2. if correct, randomly compress one modality
        # 3. if not, output the compression level
        gate_label = torch.zeros(2, 4)
        i, j = len(output_cache['audio'])-1, len(output_cache['image'])-1
        while i>=0 and j>=0:
            print(i, j)
            predict_label = self.projection(torch.cat([output_cache['audio'][i], output_cache['image'][j]], dim=-1))
            if torch.argmax(predict_label, dim=-1).cpu() == label:
                random_number = torch.rand(1)
                if random_number.item() < 0.5:
                    i -= 1
                else:
                    j -= 1
            else:
                gate_label[0, i] = 1
                gate_label[1, j] = 1
                break
        return gate_label
    def gate_train(self, audio, image, label):
        output_cache, output = self.forward(audio, image, random_threshold=(-1, -1, -1)) # get all the possibilities
        gate_label = self.label(output_cache, label) # [batch, 4 * 2]
        output, gate = self.gate(output_cache) # [batch, 4 * 2]
        loss_c = nn.functional.cross_entropy(gate, gate_label) # compression-level loss
        output = self.projection(output, dim=1)
        loss_r = nn.functional.cross_entropy(output, label) # recognition-level loss
        return loss_c, loss_r

    def forward(self, audio, image, random_threshold=(0.25, 0.33, 0.5)):
        '''
        :param audio: Modality 1
        :param image: Modality 2
        :param random_threshold: Random exit for training, set = (-1, -1 ...) if want gate training (no exit)
        :return:
        '''
        self.audio_exit = False
        self.image_exit = False
        audio, image = self.stem(audio, image)
        output_cache = {'audio': [], 'image': []}

        audio = self.audio.layer1(audio)
        early_output = self.early_exit1a(audio)
        output_cache['audio'].append(early_output)
        image = self.image.layer1(image)
        early_output = self.early_exit1b(image)
        output_cache['image'].append(early_output)
        if self.gate is None:
            self.inference_update('audio_exit', random_thres=random_threshold[0])
            self.inference_update('image_exit', random_thres=random_threshold[0])
        else:
            _, gate = self.gate(output_cache)
            self.inference_update('audio_exit', gate[0], level=1)
            self.inference_update('image_exit', gate[1], level=1)

        if not self.audio_exit:
            audio = self.audio.layer2(audio)
            early_output = self.early_exit2a(audio)
            output_cache['audio'].append(early_output)
        if not self.image_exit:
            image = self.image.layer2(image)
            early_output = self.early_exit2b(image)
            output_cache['image'].append(early_output)
        if self.gate is None:
            self.inference_update('audio_exit', random_thres=random_threshold[1])
            self.inference_update('image_exit', random_thres=random_threshold[1])
        else:
            _, gate = self.gate(output_cache)
            self.inference_update('audio_exit', gate[0], level=2)
            self.inference_update('image_exit', gate[1], level=2)

        if not self.audio_exit:
            audio = self.audio.layer3(audio)
            early_output = self.early_exit3a(audio)
            output_cache['audio'].append(early_output)
        if not self.image_exit:
            image = self.image.layer3(image)
            early_output = self.early_exit3b(image)
            output_cache['image'].append(early_output)
        if self.gate is None:
            self.inference_update('audio_exit', random_thres=random_threshold[2])
            self.inference_update('image_exit', random_thres=random_threshold[2])
        else:
            _, gate = self.gate(output_cache)
            self.inference_update('audio_exit', gate[0], level=3)
            self.inference_update('image_exit', gate[1], level=3)

        if not self.audio_exit:
            audio = self.audio.layer4(audio)
            early_output = self.early_exit4a(audio)
            output_cache['audio'].append(early_output)
        if not self.image_exit:
            image = self.image.layer4(image)
            early_output = self.early_exit4b(image)
            output_cache['image'].append(early_output)

        output = self.projection(torch.cat([output_cache['audio'][-1], output_cache['image'][-1]], dim=1))
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