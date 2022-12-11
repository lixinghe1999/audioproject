'''
Utils include converting, saving, measurement
'''
import torch
import time
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.quantization import get_default_qconfig, quantize_jit
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import copy
import onnx

def model_save_android(model, name, sample_input):
    model.eval()
    scripted_module = torch.jit.trace(model, sample_input)
    optimized_scripted_module = optimize_for_mobile(scripted_module)
    optimized_scripted_module._save_for_lite_interpreter(name + ".ptl")

def model_onnx(model, name, sample_input):
    model.eval()
    torch.save(model, name + ".pt")
    output_model_file = name + '.onnx'
    torch.onnx.export(model,  # model being run
                      sample_input,  # model input (or a tuple for multiple inputs)
                      output_model_file,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output']  # the model's output names
                      )
    onnx_model = onnx.load(output_model_file)
    onnx.checker.check_model(onnx_model)
    print('The model has been saved at: {}.onnx'.format(name))

def model_quantize_save(model, name):
    model.eval()
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {
        "": qconfig,
    }
    model_to_quantize = copy.deepcopy(model)
    prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
    print("prepared model: ", prepared_model)
    quantized_model = convert_fx(prepared_model)
    print("quantized model: ", quantized_model)
    torch.save(model.state_dict(), name + ".pth")
    torch.save(quantized_model.state_dict(), name + "_quant.pth")

def model_quantize(model):
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {
        "": qconfig,
    }
    model_prepared = prepare_fx(model, qconfig_dict)
    model_int8 = convert_fx(model_prepared)
    return model_int8

def model_speed(model, sample_input):
    t_start = time.time()
    step = 100
    with torch.no_grad():
        for i in range(step):
            model(sample_input)
    return (time.time() - t_start) / step

