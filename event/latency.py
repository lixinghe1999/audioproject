import torch
from model.dynamicvit_model import AVnet_Dynamic
from model.gate_model import AVnet_Gate
import time
import argparse
import matplotlib.pyplot as plt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='gate')
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-n', '--num', default=10, type=int)
    parser.add_argument('-e', '--exits', nargs='+', default='11 11')
    parser.add_argument('-l', '--locations', nargs='+', default='3 6 9')
    parser.add_argument('-r', '--rate', default=0.2, type=float)
    args = parser.parse_args()
    task = args.task
    device = torch.device(args.device)

    audio = torch.zeros(1, 384, 128).to(device)
    image = torch.zeros(1, 3, 224, 224).to(device)
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
        pruning_loc = [int(i) for i in args.locations.split()]
        base_rate = args.rate
        token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
        model = AVnet_Dynamic(pruning_loc=pruning_loc, token_ratio=token_ratio, pretrained=False).to(device)
        model.eval()
        with torch.no_grad():
            for i in range(num_iteration):
                if i == 1:
                    t_start = time.time()
                x, t_stamp = model(audio, image)
                plt.plot(t_stamp, c='b')
            print('latency:', (time.time() - t_start) / (num_iteration - 1))

        model = AVnet_Dynamic(pruning_loc=(), pretrained=False).to(device)
        model.eval()
        with torch.no_grad():
            for i in range(num_iteration):
                if i == 1:
                    t_start = time.time()
                x, t_stamp = model(audio, image)
                plt.plot(t_stamp, c='r')
            print('latency:', (time.time() - t_start) / (num_iteration - 1))
        plt.show()