import numpy as np
import random
import torch

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# hook function
class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException

def save_inp_oup_data(model, module, cali_data: list, store_inp=False, store_oup=False, bs: int = 32, keep_gpu: bool = True):
    """
    Save the input and output data of a layer/block
    """
    device = next(model.parameters()).device
    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    with torch.inference_mode():
        for i in range(int(cali_data.size(0) / bs)):
            try:
                _ = model(cali_data[i * bs: (i + 1) * bs].to(device))
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0].detach())
                else:
                    cached[0].append(data_saver.input_store[0].detach().cpu())
            if store_oup:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    cached[1].append(data_saver.output_store.detach().cpu())
    if store_inp:
        cached[0] = torch.cat([x for x in cached[0]])
    if store_oup:
        cached[1] = torch.cat([x for x in cached[1]])
    handle.remove()
    torch.cuda.empty_cache()
    return cached


