import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from DL.A2net import A2net


model = A2net(inference=True)

model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('candidate.pth').items()})
dummy_input = (Variable(torch.randn(1, 1, 33, 151)), Variable(torch.randn(1, 1, 264, 151)))
torch.onnx.export(model, args=dummy_input, f="candidate.onnx", input_names=["input_1", "input_2"],
        output_names=["output1"])