import utils
from model.identification import VGGM
from model.localization import DeepEar
import torch
if __name__=="__main__":
    #device = torch.device("cuda")
    device = torch.device("cpu")

    # identification = VGGM(1251).to(device)
    # sample_input = torch.rand((1, 1, 512, 300)).to(device)
    # print(utils.model_speed(model=identification, sample_input=sample_input))

    localization = DeepEar().to(device)
    Binaural = torch.rand((1, 2, 300, 512)).to(device)
    cross = torch.rand((1, 1, 100)).to(device)
    print(utils.model_speed(model=localization, sample_input=[Binaural, cross]))
