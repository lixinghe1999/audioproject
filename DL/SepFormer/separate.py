import argparse
import os
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.data_loader import MyDataset
import soundfile as sf

from model.dual_path import SepformerWrapper
from src.utils import remove_pad
import json5
import time

class SepFormer:
    def __init__(self, model_path='SepFormer/checkpoint/pretrain.pth'):
        self.model = SepformerWrapper()
        self.model.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            self.model.cuda()
    def separate_file(self, path):
        with torch.no_grad():
            data = librosa.load(path, sr=8000)[0]
            data = torch.from_numpy(np.expand_dims(data, axis=0))
            if torch.cuda.is_available():
                data = data.cuda()

            estimate_source = self.model(data)  # 将数据放入模型
            return estimate_source


def main(config):

    # model = Sepformer(N=config["model"]["sepformer"]["N"],
    #                       C=config["model"]["sepformer"]["C"],
    #                       L=config["model"]["sepformer"]["L"],
    #                       H=config["model"]["sepformer"]["H"],
    #                       K=config["model"]["sepformer"]["K"],
    #                       Global_B=config["model"]["sepformer"]["Global_B"],
    #                       Local_B=config["model"]["sepformer"]["Local_B"])
    #model.load_state_dict(torch.load(config["model_path"])["state_dict"])
    model = SepformerWrapper()
    model.load_state_dict(torch.load(config["model_path"]))

    model.eval()  # 将模型设置为校验模式

    if torch.cuda.is_available():
        model.cuda()

    # 加载数据
    eval_dataset = MyDataset(data_dir=config["validation_dataset"]["validation_dir"],
                           sr=config["validation_dataset"]["sample_rate"],
                             duration=config["validation_dataset"]["segment"])

    eval_loader = DataLoader(eval_dataset,
                           batch_size=config["validation_loader"]["batch_size"],
                           shuffle=config["validation_loader"]["shuffle"],
                           num_workers=config["validation_loader"]["num_workers"])

    os.makedirs(config["out_dir"], exist_ok=True)
    os.makedirs(config["out_dir"]+"/mix/", exist_ok=True)
    os.makedirs(config["out_dir"]+"/s1/", exist_ok=True)
    os.makedirs(config["out_dir"]+"/s2/", exist_ok=True)

    with torch.no_grad():

        for (i, data) in enumerate(eval_loader):

            print("{}-th Batch Data Start Generate".format(i))

            start_time = time.time()

            mixture, mix_lengths, _ = data

            if torch.cuda.is_available():

                mixture = mixture.cuda()

                mix_lengths = mix_lengths.cuda()

            estimate_source = model(mixture)  # 将数据放入模型
            # Remove padding and flat
            flat_estimate = remove_pad(estimate_source.permute(0, 2, 1), mix_lengths)

            mixture = remove_pad(mixture, mix_lengths)

            for j in range(len(mixture)):
                filename = os.path.join(config["out_dir"]+"/mix/", str(i) + '_' + str(j))
                sf.write(filename + '.wav', mixture[j], samplerate=config["sample_rate"])
                C = flat_estimate[j].shape[0]
                for c in range(C):
                    if c == 0:
                        filename = os.path.join(config["out_dir"] + "/s1/", str(i) + '_' + str(j))
                        sf.write(filename + '.wav', flat_estimate[j][c], samplerate=config["sample_rate"])
                    elif c == 1:
                        filename = os.path.join(config["out_dir"] + "/s2/", str(i) + '_' + str(j))
                        sf.write(filename + '.wav', flat_estimate[j][c], samplerate=config["sample_rate"])

            end_time = time.time()

            run_time = end_time - start_time

            print("Elapsed Time: {} s".format(run_time))

        print("Data Generation Completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Separation")

    parser.add_argument("-C",
                        "--configuration",
                        default="./config/test/separate.json5",
                        type=str,
                        help="Configuration (*.json).")

    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))

    main(configuration)
