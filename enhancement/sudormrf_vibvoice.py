from sudormrf import GlobLN, UConvBlock
import math
import torch.nn as nn
import torch
class sudormrf(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=8,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=320,
                 num_sources=1):
        super(sudormrf, self).__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=enc_num_basis,
            out_channels=out_channels,
            kernel_size=1)

        # Separation module
        self.sm = nn.ModuleList([UConvBlock(out_channels=out_channels,
                       in_channels=in_channels,
                       upsampling_depth=upsampling_depth)
            for _ in range(num_blocks)])

        mask_conv = nn.Conv1d(out_channels, num_sources * enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        self.mask_nl_class = nn.ReLU()
    def forward(self, x, acc=None):
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        std = torch.std(x, dim=(1, 2), keepdim=True)
        x = (x - mean) / (std + 1e-9)

        s = x.clone()
        x = self.ln(x)
        x = self.bottleneck(x)
        for m in self.sm:
            x = m(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x.squeeze(1) * s
        # Back end
        x = (x * std) + mean
        return x

if __name__ == "__main__":
    model = sudormrf()
    dummy_input = torch.rand(1, 320, 160)
    estimated_sources = model(dummy_input)
    print(estimated_sources.shape)