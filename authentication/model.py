
import yaml
import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, time_dim, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.LayerNorm([n_outputs, time_dim])
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.LayerNorm([n_outputs, time_dim])
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.norm1, self.relu1,
                                 self.conv2, self.chomp2, self.norm2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        net = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(net + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, time_dim, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, time_dim, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        #self.attention = nn.Linear(out_channels, out_channels)
        #self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.linear(x)
        #x = self.norm(x)
        #x = x * self.attention(x)
        return x + self.linear(x)
class MLP_attention(nn.Module):
    def __init__(self, num_inputs, num_channels):
        super(MLP_attention, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(MLP(in_channels, out_channels))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)


class Model(nn.Module):
    def __init__(self, model_paras):
        super(Model, self).__init__()
        self.model_paras = model_paras
        paras = model_paras
        # self.encoder = TemporalConvNet(paras['input_features'], paras['channels'],
        #                                paras['time_dim'], paras['kernel_size'], dropout=paras['drop_out'])
        # self.decoder = TemporalConvNet(paras['channels'][-1], paras['re_channels'],paras['time_dim'],
        #                                 paras['kernel_size'], dropout=paras['drop_out'])
        self.mlp = MLP_attention(paras['input_features'], paras['channels'])
        self.fls_layer = nn.Linear(paras['channels'][-1], 15)
        #self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, data):
        # data = self.encoder(data)
        # data = self.decoder(data)
        # data = torch.flatten(data, start_dim=1)
        data = self.mlp(data)
        data = self.fls_layer(data)
        #data = self.softmax(data)
        return data
if __name__ == '__main__':
    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
    with open('model.yaml', 'r') as file:
        config = yaml.safe_load(file)
    model = Model(config['model_params']).cuda()
    #print('total num params = {:.2f}M'.format(get_n_params(model) / 1.0e6))
    tensor_example = torch.zeros((2, 1, 66)).cuda()
    print(model(tensor_example).shape)
#   torch.save(model.state_dict(), 'model.pth')
