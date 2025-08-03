from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import (UniformQuantizer_DeMax, UniformQuantizer)

GELU_MIN =  -0.16997122764587402

class QuantConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    """
    def __init__(self,   
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                input_quant_params={},
                weight_quant_params={}):
        super(QuantConv2d, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)

        input_quant_params_conv = deepcopy(input_quant_params)
        input_quant_params_conv['n_bits'] = 8
        self.input_quantizer = UniformQuantizer(**input_quant_params_conv)
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)
        self.use_input_quant = False
        self.use_weight_quant = False


    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def set_initquant_state(self, inited=False):
        self.input_quantizer.inited= inited
        self.weight_quantizer.inited = inited


    def forward(self, x):
        """
        using quantized weights to forward input x
        """
        if self.use_input_quant:
            x = self.input_quantizer(x)

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight

        out = F.conv2d(
            x, 
            w, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )
        return out


class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """

    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 norm_quant = False):
        super(QuantLinear, self).__init__(in_features, out_features)
        self.input_quantizer = UniformQuantizer(**input_quant_params)
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)
        self.weight_quantizer_gelu = UniformQuantizer(**weight_quant_params)
        if norm_quant:
            self.input_quantizer_obs = UniformQuantizer(**input_quant_params)
            obsever_quant_params = deepcopy(weight_quant_params)
            obsever_quant_params['symmetric'] = True
            self.observer = UniformQuantizer(**obsever_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

        ### for normquant
        self.bias_x = 0
        self.norm_quant = norm_quant
        self.alph = 0


    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def set_initquant_state(self, inited=False):
        self.input_quantizer.inited= inited
        self.weight_quantizer.inited = inited


    def forward(self, x):
        """
        using quantized weights to forward input x
        """

        if self.use_input_quant:

            if self.norm_quant:
                if len(x.shape) == 2:
                    raise NotImplementedError
                if self.input_quantizer.inited == False:

                    _ = self.observer(self.weight.data.transpose(0,1).contiguous()) if self.observer.inited == False else None
                    weight_delta = self.observer.delta.reshape(-1)
                    weight_delta = weight_delta / torch.mean(weight_delta)
                    _ = self.input_quantizer_obs(x) if self.input_quantizer_obs.inited == False else None
                    act_delta = self.input_quantizer_obs.delta.reshape(-1)
                    act_zero_point = self.input_quantizer_obs.zero_point.reshape(-1)
                    act_min = -act_zero_point * act_delta
                    
                    target_delta = torch.mean(act_delta)
                    target_zero_point = torch.mean(act_zero_point)
                    target_min = -target_zero_point * target_delta
                    self.r = (act_delta / target_delta)**(1- self.alph) / (weight_delta)**self.alph

                    self.r = torch.ones_like(self.r).to(x.device) if self.alph == -1 else self.r
                    self.b = act_min / self.r - target_min
                    self.b = torch.zeros_like(self.b).to(x.device) if self.alph == -1 else self.b
                    self.input_quantizer.delta = target_delta
                    self.input_quantizer.zero_point = target_zero_point
                    bias = torch.mm(self.weight * self.r.reshape(-1), self.b.reshape(-1,1)).reshape(-1)
                    self.bias_x = bias
                    self.input_quantizer.inited = True if self.alph == 0  else False
                    self.input_quantizer.inited =  False
                    self.input_quantizer.channel_wise = False
                    self.weight_quantizer.inited = False

                x = self.input_quantizer(x/self.r - self.b)
                w = self.weight_quantizer(self.weight*self.r.reshape(-1))
            else:
                x = self.input_quantizer(x)

        
        if self.use_weight_quant:
            try:
                w
            except NameError:
                if self.norm_quant:
                    w = self.weight_quantizer(self.weight*self.r.reshape(-1))
                else:
                    w = self.weight_quantizer(self.weight)
        else:
            w = self.weight  
        out = F.linear(x, weight=w, bias=self.bias.data + self.bias_x) if self.bias is not None else F.linear(x, weight=w, bias=torch.tensor(self.bias_x, dtype=torch.float32).to(x.device))
        return out
        
class QuantLinear_no_b(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """

    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 norm_quant = False):
        super(QuantLinear_no_b, self).__init__(in_features, out_features)

        self.input_quantizer = UniformQuantizer(**input_quant_params)
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)
        self.weight_quantizer_gelu = UniformQuantizer(**weight_quant_params)
        if norm_quant:
            obsever_quant_params = deepcopy(input_quant_params)
            self.input_quantizer_obs = UniformQuantizer(**obsever_quant_params)
            obsever_quant_params = deepcopy(weight_quant_params)
            obsever_quant_params['symmetric'] = True
            self.observer = UniformQuantizer(**obsever_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

        self.bias_x = 0
        self.norm_quant = norm_quant
        self.alph = -1

    def __repr__(self):
        s = super(QuantLinear_no_b, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def set_initquant_state(self, inited=False):
        self.input_quantizer.inited= inited
        self.weight_quantizer.inited = inited


    def forward(self, x):
        """
        using quantized weights to forward input x
        """

        if self.use_input_quant:

            if self.norm_quant:
                if len(x.shape) == 2:
                    raise NotImplementedError               
                if self.input_quantizer.inited == False:
                    if abs(self.alph + 1) < 1e-5:
                        self.r = torch.ones(x.shape[2]).to(x.device)
                        self.b = 0
                        self.bias_x = 0
                    else:
                        _ = self.observer(self.weight.data.transpose(0,1).contiguous()) if self.observer.inited == False else None
                        weight_delta = self.observer.delta.reshape(-1)
                        weight_delta = weight_delta / torch.mean(weight_delta)
                        if self.input_quantizer_obs.n_bits == 6:
                            act_delta = x.reshape(-1,x.shape[-1]).abs().max(dim = 0)[0].reshape(-1)
                        else:
                            _ = self.input_quantizer_obs(x)
                            act_delta = (_.reshape(-1,x.shape[-1])).abs().max(dim = 0)[0].reshape(-1)

                        act_delta = act_delta.clamp(0, None)
                        target_delta = torch.mean(act_delta) 
                        act_delta = act_delta.clamp(target_delta, None)
                        self.r = (act_delta / target_delta)**(1- self.alph) / (weight_delta)**self.alph
                        self.r = self.r.clamp(1, None)
                        self.r = torch.ones_like(self.r).to(x.device) if abs(self.alph + 1) < 1e-5 else self.r
                        self.b = 0
                        self.bias_x = 0

                    self.input_quantizer.inited =  False
                    self.input_quantizer.channel_wise = False
                    self.weight_quantizer.inited = False

                if abs(self.alph + 1) < 1e-5:
                    x = self.input_quantizer(x)
                else:  
                    x = x/self.r - self.b
                    x = self.input_quantizer(x)
                    w = self.weight_quantizer(self.weight*self.r.reshape(-1))
 
            else:
                x = self.input_quantizer(x)

        if self.use_weight_quant:
            try:
                w
            except NameError:
                if self.norm_quant:
                    w = self.weight_quantizer(self.weight*self.r.reshape(-1))
                else:
                    w = self.weight_quantizer(self.weight)
        else:
            w = self.weight  
        out = F.linear(x, weight=w, bias=self.bias.data + self.bias_x) if self.bias is not None else F.linear(x, weight=w, bias=torch.tensor(self.bias_x, dtype=torch.float32).to(x.device))
        return out
    
class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 input_quant_params={}):
        super(QuantMatMul, self).__init__()
        input_quant_params_matmul = deepcopy(input_quant_params)
        if 'demax_quant' in input_quant_params_matmul:
            input_quant_params_matmul.pop('demax_quant')
            self.quantizer_A = UniformQuantizer_DeMax(**input_quant_params_matmul)
        else:
            self.quantizer_A = UniformQuantizer(**input_quant_params_matmul)
        self.quantizer_B = UniformQuantizer(**input_quant_params_matmul)
        self.use_quantizer_A = True
        self.use_input_quant = False

    def __repr__(self):
        s = super(QuantMatMul, self).__repr__()
        s = "(" + s + "input_quant={})".format(self.use_input_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant

    def set_initquant_state(self, inited=False):
        self.quantizer_A.inited= inited
        self.quantizer_B.inited = inited

    def forward(self, A, B):
        if self.use_input_quant:
            if  self.use_quantizer_A:
                A = self.quantizer_A(A)
                B = self.quantizer_B(B)
            else:
                A = A
                B = self.quantizer_B(B)
        out = A @ B
        return out
