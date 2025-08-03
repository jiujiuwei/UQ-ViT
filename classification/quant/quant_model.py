from copy import deepcopy
import torch.nn as nn
from utils.build_model import MatMul
from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul, QuantLinear_no_b


def quant_model(model, input_quant_params={}, weight_quant_params={}):

    input_quant_params_matmul1 = deepcopy(input_quant_params)
    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    input_quant_params_matmul2['demax_quant'] = True
    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True

    module_dict={}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        
        if isinstance(m, nn.Conv2d):
            # Embedding Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantConv2d(
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                m.stride,
                m.padding,
                m.dilation,
                m.groups,
                m.bias is not None,
                input_quant_params,
                weight_quant_params
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            if 'qkv' in name or 'fc1' in name or 'reduction' in name :
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, norm_quant = True)
            elif 'fc2' in name or ("proj" in name ):
                new_m = QuantLinear_no_b(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, norm_quant = True)
            else:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params,)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx
            if 'matmul2' in name:
                new_m = QuantMatMul(input_quant_params_matmul2)
            else:
                new_m = QuantMatMul(input_quant_params_matmul1)
            setattr(father_module, name[idx:], new_m)

    return model


def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul, QuantLinear_no_b)):
            m.set_quant_state(input_quant, weight_quant)


def set_initquant_state(model, inited=False):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul, QuantLinear_no_b)):
            m.set_initquant_state(inited = inited)

