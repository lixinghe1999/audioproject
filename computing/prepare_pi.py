import utils
from identification.vggm import VGGM
import torch
if __name__ == "__main__":
    device = torch.device("cpu")
    identification = VGGM(1251).to(device)
    sample_input = torch.rand((1, 1, 512, 300)).to(device)
    utils.model_quantize_save(identification, 'identification')
    utils.model_onnx(identification, 'identification', sample_input)