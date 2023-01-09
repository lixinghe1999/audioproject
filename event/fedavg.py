import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import yaml
import shutil
from pathlib import Path
from model import AudioCLIP
from utils.datasets.esc50 import ESC50
from utils.datasets.split_dataset import split_dataset, split_dataset_type, split_type, split_type_random

from fed import Fed_client, Fed_server
def deterministic(seed):
    """
    Make the experiment reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="protocols/fedavg.yaml", type=str)
    parser.add_argument("--exp_name", default="fedavg", type=str, help="exp name")
    parser.add_argument("--non_iidness", default=1, type=int, help="non-iid degree of distributed data")
    parser.add_argument('--tao_ratio', type=float, default=2, choices=[0.5, 1, 2, 4])
    # optional params
    parser.add_argument('--seed', default=1, type=int, help="using fixed random seed")
    parser.add_argument("--work_dir", default="./runs_exp", type=str, help="output dir")
    # unused params
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()

    # random seed
    if args.seed is not None:
        deterministic(args.seed)

    # config
    log_dir = f'{args.work_dir}/{args.exp_name}'  # _{args.tao_ratio}'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(args.cfg, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    num_client = config["fl_opt"]["num_clients"]
    gpu_list = [0, 1, 2, 3][: torch.cuda.device_count()]
    gpu_idx = [i % len(gpu_list) for i in range(num_client)]
    config.update({"device": torch.device('cpu' if torch.cuda.is_available() else 'cpu')})  # init_network
    config.update({"device_client": [torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu') for i in
                                     gpu_idx]})  # client device\
    print(f"gpu of clients: {gpu_idx}")

    # init model and criterion on cpu
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    model = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to('cpu')
    SAMPLE_RATE = 44100
    train_dataset = ESC50('../dataset/ESC50', fold=1, train=True, sample_rate=SAMPLE_RATE, few_shot=None)
    test_dataset = ESC50('../dataset/ESC50', fold=1, train=False, sample_rate=SAMPLE_RATE)
    num_users = num_client
    train_dataset_list, type_list = split_dataset(train_dataset, num_users)
    test_dataset_list = split_dataset_type(test_dataset, type_list)

    # multi-process setup
    import multiprocessing as mp
    if torch.cuda.is_available():
        mp.set_start_method('spawn')
    process_list = []
    state_list = mp.Manager().list([0 for i in range(num_client)])
    model_list = mp.Manager().list([None for i in range(num_client)])

    # FL class for servrer
    fed = Fed_server(
        model, config, train_dataset, test_dataset, state_list, model_list)
    aggre_mode = config["fl_opt"]["aggregation"]
    frac = config["fl_opt"]["frac"]
    client_per_round = max(int(frac * fed.num_clients), 1)

    # FL class for clients
    for i in range(num_client):
        client_i_train = train_dataset_list[i]
        client_i_test = test_dataset_list[i]
        p = Fed_client(
            model, config, client_i_train, client_i_test, state_list, model_list, idx=i
        )
        p.daemon = True
        p.start()
        process_list.append(p)

    """"
    FL starts 
    """
    # training mode
    if not args.test:
        best_acc = 0

        # FL rounds
        for round_i in range(config["fl_opt"]["rounds"]):

            # select users
            selected_idx = np.random.choice(
                range(fed.num_clients), client_per_round, replace=False)
            # select classes and num_of_samples per class
            selected_cls = []
            for i in selected_idx:
                selected_cls += list(cls_per_client[i])
            print_write([f'\n Round: {round_i}, selected clients: {selected_idx}'], log_file)
            # print_write([f'selected cls: {set(selected_cls)}'], log_file)

            # train and aggregate
            fed.local_train(selected_idx)
            fed.aggregation(selected_idx, aggre_mode)

            # evaluate
            train_loss_per_cls, train_acc_per_cls, test_loss_per_cls, test_acc_per_cls = fed.evaluate_global()
            train_loss = train_loss_per_cls.mean()
            train_acc = train_acc_per_cls.mean()

            # testset (e.g., Cifar-100) is equally distributed among classes
            test_loss_mean = test_loss_per_cls.mean()
            test_acc_mean = test_acc_per_cls.mean()

            # logging
            kd_loss = np.array(fed.losses_kd)[selected_idx].mean()
            cls_loss = np.array(fed.losses_cls)[selected_idx].mean()
            np.set_printoptions(precision=3)
            print_write(["cls_loss, kd_loss, train_loss, test_loss, train_acc, test_acc: ", \
                         [cls_loss, kd_loss, train_loss, test_loss_mean, train_acc, test_acc_mean]], log_file)
            print_write(["per_cls_acc (train): ", train_acc_per_cls], log_file)
            print_write(["per_cls_acc: ", test_acc_per_cls], log_file)

            write_summary(
                tensorboard, split='train', step=round_i, kd_loss=kd_loss,
                cls_loss=cls_loss, loss=train_loss, acc=train_acc,
                cls0_acc=train_acc_per_cls[0], cls2_acc=train_acc_per_cls[1], cls3_acc=train_acc_per_cls[2],
                cls4_acc=train_acc_per_cls[3], cls5_acc=train_acc_per_cls[4], cls6_acc=train_acc_per_cls[5])
            write_summary(
                tensorboard, split='val', step=round_i, loss=test_loss_mean, acc=test_acc_mean,
                cls0_acc=test_acc_per_cls[0], cls2_acc=test_acc_per_cls[1], cls3_acc=test_acc_per_cls[2],
                cls4_acc=test_acc_per_cls[3], cls5_acc=test_acc_per_cls[4], cls6_acc=test_acc_per_cls[5]
            )

            # save ckpts
            if test_acc_mean > best_acc:
                ckpt = {'round_i': round_i, 'model': fed.server_network}
                ckpt_name = f"{log_dir}/best.pth"
                torch.save(ckpt, ckpt_name)
                best_acc = test_acc_mean
                print_write([f"best round: {round_i}, accuracy: {test_acc_mean * 100}"], log_file)
                # del ckpt

    # Currently do not support test mode. Use `evaluate.py` instead.
    else:
        pass

    for i in process_list:
        p.join()
