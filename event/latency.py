import torch
from model.dynamicvit_model import AVnet_Dynamic
from model.gate_model import AVnet_Gate
from model.vit_model import VisionTransformerDiffPruning
import time
import argparse
from fvcore.nn import FlopCountAnalysis
import numpy as np
def rfft_flop_jit(inputs, outputs):
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops
def calc_flops(model, input, show_details=False, ratios=None):
    with torch.no_grad():
        model.default_ratio = ratios
        fca1 = FlopCountAnalysis(model, input)
        handlers = {
            'aten::fft_rfft2': rfft_flop_jit,
            'aten::fft_irfft2': rfft_flop_jit,
        }
        fca1.set_op_handle(**handlers)
        flops1 = fca1.total()
        if show_details:
            print(fca1.by_module())
        print("#### GFLOPs: {} for ratio {}".format(flops1 / 1e9, ratios))
    return flops1 / 1e9

@torch.no_grad()
def throughput(images, model):
    model.eval()
    batch_size = images[0].shape[0]
    for i in range(50):
        model(*images)
    torch.cuda.synchronize()
    tic1 = time.time()
    for i in range(30):
        model(*images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    MB = 1024.0 * 1024.0
    print('memory:', torch.cuda.max_memory_allocated() / MB)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='gate')
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-b', '--batch', default=1, type=int)
    parser.add_argument('-n', '--num', default=10, type=int)
    parser.add_argument('-e', '--exits', nargs='+', default='11 11')
    parser.add_argument('-l', '--locations', nargs='+', default='3 6 9')
    parser.add_argument('-r', '--rate', default=0.7, type=float)
    args = parser.parse_args()
    task = args.task
    device = torch.device(args.device)

    audio = torch.zeros(args.batch, 384, 128).to(device, non_blocking=True)
    image = torch.zeros(args.batch, 3, 224, 224).to(device, non_blocking=True)
    if args.device == 'cuda' and args.task == 'dynamic':
        assert (args.batch == 1), "Right now for GPU inference -> batch should equal to 1"
    pruning_loc = [int(i) for i in args.locations.split()]
    base_rate = args.rate
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
    num_iteration = args.num
    if task == 'gate':
        exits = [int(i) for i in args.exits.split()]
        model = AVnet_Gate().to(device)
        model.eval()
        with torch.no_grad():
            for i in range(num_iteration):
                if i == 1:
                    t_start = time.time()
                model(audio, image, mode=torch.tensor(exits))
            print('latency:', (time.time() - t_start) / (num_iteration - 1))
    elif task == 'dynamic':

        model = AVnet_Dynamic(pruning_loc=pruning_loc, token_ratio=token_ratio, pretrained=False).to(device)
        throughput([audio, image], model)
        # calc_flops(model, (audio, image))

        model = AVnet_Dynamic(pruning_loc=(), pretrained=False).to(device)
        throughput([audio, image], model)
        # calc_flops(model, (audio, image))

    elif task == 'dynamicvit':
        config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                      pruning_loc=pruning_loc, token_ratio=token_ratio)
        model = VisionTransformerDiffPruning(**config).to(device)
        throughput([image], model)

        config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                      pruning_loc=(), token_ratio=token_ratio)
        model = VisionTransformerDiffPruning(**config).to(device)
        throughput([image], model)