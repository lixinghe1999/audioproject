import utils
import onnxruntime as ort
import numpy as np
import torch
import time
from model.identification import VGGM
from preprocess import preprocess

def inference_onnx(onnx_model, sample_input):
    ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
    t_start = time.time()
    step = 30
    for i in range(step):
        spec = preprocess(sample_input)
        spec = np.expand_dims(spec, (0, 1))
        output = ort_session.run(
            None,
            {"input": spec.astype(np.float32)},
        )
    fps = step / (time.time() - t_start)
    return fps

def inference_torch(quant_model, sample_input):
    torch.backends.quantized.engine = 'qnnpack'
    model = VGGM(1251)
    model_int8 = utils.model_quantize(model)
    ckpt = torch.load(quant_model)
    model_int8.load_state_dict(ckpt)
    model_int8.eval()
    #model_int8 = torch.jit.script(model_int8)
    t_start = time.time()
    step = 30
    for i in range(step):
        spec = preprocess(sample_input)
        spec = np.expand_dims(spec, (0, 1))
        spec = torch.from_numpy(spec).type(torch.float)
        output = model_int8(spec)
    fps = step / (time.time() - t_start)
    return fps
if __name__ == "__main__":

    # sample_input = np.random.random((48320))
    # torch.set_num_threads(1)
    # print('identification FPS onnx', inference_onnx('identification.onnx', sample_input))
    # print('identification FPS quantized torch', inference_torch('identification_quant.pth', sample_input))

    sample_input = np.random.random((48320))
    torch.set_num_threads(1)
    print('localization FPS onnx', inference_onnx('localization.onnx', sample_input))
    #print('localization FPS quantized torch', inference_torch('identification_quant.pth', sample_input))