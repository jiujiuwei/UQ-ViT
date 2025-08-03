import numpy as np
import torch
import torch.nn as nn
GELU_MIN =  -0.16997122764587402

def calculate_quantiles(x, pct):
    """
    Calculate the quantiles of a tensor
    """
    try:
        new_max = torch.quantile(x.reshape(-1), pct, interpolation='nearest')
        new_min = torch.quantile(x.reshape(-1), 1 - pct, interpolation='nearest')

    except RuntimeError as e:
        try:
            k = round((x.reshape(-1)).numel() * (1 - pct) + 1)
            topknums = torch.topk(x.reshape(-1), k, largest=True, sorted = False).values
            new_max = topknums.min()
            topknums = torch.topk(x.reshape(-1), k, largest=False, sorted = False).values
            new_min = topknums.max()
        except RuntimeError as e:
 
            x_cpu = x.cpu().detach().numpy() if x.requires_grad else x.cpu().numpy()
            new_max_np = np.percentile(x_cpu, pct * 100, interpolation='nearest')
            new_min_np = np.percentile(x_cpu, (1 - pct) * 100, interpolation='nearest')
            # new_max_np = np.percentile(x_cpu, pct * 100)
            # new_min_np = np.percentile(x_cpu, (1 - pct) * 100)
            new_max = torch.tensor(new_max_np, device=x.device, dtype=torch.float32)
            new_min = torch.tensor(new_min_np, device=x.device, dtype=torch.float32)
    
    return new_max, new_min

def calculate_quantiles_max(x, pct):
    """
    Calculate the quantiles of a tensor
    """
    try:
        new_max = torch.quantile(x.reshape(-1), pct, interpolation='nearest')
    except RuntimeError as e:
        try:
            k = round((x.reshape(-1)).numel() * (1 - pct) + 1)
            topknums = torch.topk(x.reshape(-1), k, largest=True, sorted = False).values
            new_max = topknums.min()

        except RuntimeError as e:

            x_cpu = x.cpu().detach().numpy() if x.requires_grad else x.cpu().numpy()
            new_max_np = np.percentile(x_cpu, pct * 100, interpolation='nearest')
            # new_max_np = np.percentile(x_cpu, pct * 100)
            # new_min_np = np.percentile(x_cpu, (1 - pct) * 100)
            new_max = torch.tensor(new_max_np, device=x.device, dtype=torch.float32)
           
    return new_max

def calculate_quantiles_min(x, pct):
    """
    Calculate the quantiles of a tensor
    """

    try:
        new_min = torch.quantile(x.reshape(-1), 1 - pct, interpolation='nearest')

    except RuntimeError as e:
        try:
            k = round((x.reshape(-1)).numel() * (1 - pct) + 1)
            topknums = torch.topk(x.reshape(-1), k, largest=False, sorted = False).values
            new_min = topknums.max()
        except RuntimeError as e:

            x_cpu = x.cpu().detach().numpy() if x.requires_grad else x.cpu().numpy()
            new_min_np = np.percentile(x_cpu, (1 - pct) * 100, interpolation='nearest')
            new_min = torch.tensor(new_min_np, device=x.device, dtype=torch.float32)
    return new_min

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()
    
class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, symmetric = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.symmetric = symmetric

    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s
    
    def forward(self, x: torch.Tensor):
        if self.symmetric:
            return self.forward_symmetric(x)
        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        x_dequant = (torch.clamp(torch.round(x / self.delta) + self.zero_point, 0, self.n_levels - 1) - self.zero_point) * self.delta
        return x_dequant
    

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x
            x_max = x_clone.max()
            best_score = 1e+10

            pcts =  [0.999, 0.9999, 0.99999]
            for pct in pcts: #
                new_max, new_min = calculate_quantiles(x, pct)
                if (new_max - new_min) < 1e-5:
                    new_max = 0
                    new_min = torch.tensor(GELU_MIN).to(x.device)

                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()
        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        x_float_q = (torch.clamp(torch.round(x / delta) + zero_point, 0, self.n_levels - 1) - zero_point) * delta
        return x_float_q
        
    def forward_symmetric(self, x):
        if self.inited is False:
            self.delta = self.init_quantization_scale_symmetric(x, self.channel_wise, None)
            self.inited = True
        x_dequant = torch.clamp(torch.round(x / self.delta), -self.n_levels/2, self.n_levels/2 - 1) * self.delta
        return x_dequant

    def init_quantization_scale_symmetric(self, x: torch.Tensor, channel_wise: bool = False, head_num = None):
        delta  = None
        if channel_wise:
            x_clone = x
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c] = self.init_quantization_scale_symmetric(x_clone[:,:,c], channel_wise=False)
                else:
                    delta[c] = self.init_quantization_scale_symmetric(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)

            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)

            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)

            else:
                raise NotImplementedError
        else:
            x_clone = x
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            for pct in [0.999, 0.9999, 0.99999]:         
                new_max, new_min = calculate_quantiles(x, pct)
                abs_m = torch.max(new_max.abs(), new_min.abs())
                x_q = self.quantize_symmetric(x_clone, abs_m)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = 2*abs_m / (2 ** self.n_bits - 1)
        return delta
    
    def quantize_symmetric(self, x, abs_m):
        delta = 2*abs_m / (2 ** self.n_bits - 1)
        x_float_q = torch.clamp(torch.round(x / delta), -self.n_levels/2, self.n_levels/2 - 1) * delta
        return x_float_q



class UniformQuantizer_DeMax(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(UniformQuantizer_DeMax, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.quantizer = UniformQuantizer(n_bits=self.n_bits, channel_wise=False)

    def forward(self, x: torch.Tensor, x1 =None):
        x_dequant = self.init_quantization_uniform(x)
        return x_dequant

    def init_quantization_uniform(self, x):
        B,H,N,_ = x.shape
        self.zero_point = 0
        max_values, max_indices = torch.max(x, dim=3, keepdim=True)
        threshold_mask = (max_values >= 0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(3, max_indices, threshold_mask)
        max_values = max_values.expand_as(x)
        sub_x = mask.float() * max_values
        x = x - sub_x
        max_values_1, max_indices_1 = torch.max(x, dim=3, keepdim=True)
        self.delta = max_values_1/(self.n_levels - 1)
        x_dequant = (torch.clamp(torch.round(x / self.delta) + self.zero_point, 0, self.n_levels - 1) - self.zero_point) * self.delta + sub_x
        return x_dequant
    