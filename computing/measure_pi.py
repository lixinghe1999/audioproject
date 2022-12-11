from PIL import Image
import imageio
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import os
import time

def inference_onnx(onnx_model, sample_input):
    ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
    t_start = time.time()
    step = 100
    for i in range(100):
        outputs = ort_session.run(
            None,
            {"input": sample_input.astype(np.float32)},
        )
    fps = (time.time() - t_start) / step
    return fps
if __name__ == "__main__":
    sample_input = np.random.random((1, 1, 512, 300))
    print(inference_onnx('identification.onnx', sample_input))
