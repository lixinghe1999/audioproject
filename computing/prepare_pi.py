import utils
from model.identification import VGGM
from model.localization import DeepEar
import torch
if __name__ == "__main__":
    device = torch.device("cpu")

    # identification = VGGM(1251).to(device)
    # sample_input = torch.rand((1, 1, 512, 300)).to(device)
    # utils.model_quantize_save(identification, 'identification')
    # utils.model_onnx(identification, 'identification', sample_input)

    localization = DeepEar()
    sample_input = torch.rand((1, 2, 300, 512))
    #utils.model_quantize_save(identification, 'identification')
    utils.model_onnx(localization, 'localization', sample_input)