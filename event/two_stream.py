import torch
import torch.nn as nn


def run(iters=10):
    device = torch.device(0)

    s1 = torch.cuda.Stream(device=device)
    s2 = torch.cuda.Stream(device=device)
    x = torch.rand(size=(1024 * 4, 1024 * 4)).to(device)
    w1 = torch.rand(size=(1024 * 4, 1024 * 4)).to(device)
    w2 = torch.rand(size=(1024 * 4, 1024 * 4)).to(device)

    for i in range(iters):
        torch.cuda.nvtx.range_push('iter{}'.format(i))

        with torch.cuda.stream(s1):
            out1 = x.matmul(w1)

        with torch.cuda.stream(s2):
            out2 = x.matmul(w2)

        torch.cuda.nvtx.range_pop()


if __name__ == '__main__':
    # warmup
    run()
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.cuda.cudart().cudaProfilerStop()