import utils
from identification.vggm import VGGM
import torch
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identification = VGGM(1251).to(device)
    sample_input = torch.rand((1, 1, 512, 300)).to(device)
    print(utils.model_speed(model=identification, sample_input=sample_input))