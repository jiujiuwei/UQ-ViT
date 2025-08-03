import copy
import argparse
import time
import torch
import torch.nn as nn
from quant import *
from tqdm import tqdm
from utils import *
from timm.models.swin_transformer import PatchMerging, SwinTransformerBlock
from timm.models.vision_transformer import Block, PatchEmbed

def get_args_parser():
    parser = argparse.ArgumentParser(description="UQ-ViT", add_help=False)
    parser.add_argument("--model", default="deit_tiny",
                        choices=['vit_small', 'vit_base',
                            'deit_tiny', 'deit_small', 'deit_base', 
                            'swin_tiny', 'swin_small'],
                        help="model")
    parser.add_argument('--dataset', default="datasets/imagenet/",
                        help='path to dataset')
    parser.add_argument("--calib-batchsize", default=512,
                        type=int, help="batchsize of calibrate set")
    parser.add_argument("--calibrate", default=512,
                        type=int, help="num of calibrate set")   
    parser.add_argument("--val-batchsize", default=512,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="number of data loading workers (default: 16)")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--print-freq", default=100,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed")

    parser.add_argument('--w_bits', default=4,
                        type=int, help='bit-precision of weights')
    parser.add_argument('--a_bits', default=4,
                        type=int, help='bit-precision of activation')

    return parser



def search_alph(alph_values, child_module, input_data, output_data, name, search_module, modules):
    best_score = 1e+10
    best_alph = 0
    for module in modules:
        set_quant_state(module, input_quant=True, weight_quant=True)

    with torch.inference_mode():
        for alph in alph_values:
            # torch.cuda.empty_cache()
            search_module.alph = alph
            for module in modules:
                set_initquant_state(module, False)
            output = child_module(input_data)
            score = lp_loss(output, output_data, p=2, reduction='all')
            if score < best_score:
                best_score = score
                best_alph = alph
        search_module.alph = best_alph

        if (len(alph_values) > 1):
            for module in modules:
                set_initquant_state(module, False)
            output = child_module(input_data)  
            score = lp_loss(output, output_data, p=2, reduction='all')  
        else:
            pass
        if (score - best_score).abs() < 1e-5:
            print("name: ", name, "best_alph: ", best_alph, "best_score: ", best_score)  
        else:
            print("error")
            print("name: ", name, "best_alph: ", best_alph, "best_score: ", best_score)   
            
        for module in modules:
            set_initquant_state(module, True)

def main():
    print(args)
    seed(args.seed)

    model_zoo = {
        'vit_tiny' : 'deit_tiny_patch16_224',
        'vit_small' : 'vit_small_patch16_224',
        'vit_base' : 'vit_base_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base': 'swin_base_patch4_window7_224',
    }
    
    device = torch.device(args.device)
    print('Building model ...')
    model = build_model(model_zoo[args.model])
    model.to(device)
    model.eval()
    # Build criterion
    criterion = nn.CrossEntropyLoss().to(device)
    # Build dataloader
    print('Building dataloader ...')
    train_loader, val_loader = build_dataset(args)
    calib_dataset = []
    for data, target in train_loader:
        calib_data = data
        calib_dataset.append([calib_data.to(device), target.to(device)])
        if len(calib_dataset) * args.calib_batchsize >= args.calibrate:
            break

    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model(copy.deepcopy(model), input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device)
    q_model.eval()
    
    @torch.inference_mode()
    def recon_model(module: nn.Module, fp_module: nn.Module, name_module = None):

        for name, child_module in module.named_children():

            if isinstance(child_module, (PatchEmbed, )):
                input_data, _ = save_inp_oup_data(q_model, child_module, calib_dataset[0][0], store_inp=True, store_oup=False, bs=args.calib_batchsize)
                set_quant_state(child_module, input_quant=True, weight_quant=True)
                output_data = child_module(input_data)

            elif isinstance(child_module, (Block, SwinTransformerBlock)):
                input_data, _ = save_inp_oup_data(q_model, child_module, calib_dataset[0][0], store_inp=True, store_oup=False, bs=args.calib_batchsize)
                _, output_data = save_inp_oup_data(model, getattr(fp_module, name), calib_dataset[0][0], store_inp=False, store_oup=True, bs=args.calib_batchsize)
    
                attn = child_module.attn
                qkv = attn.qkv
                proj = attn.proj
                matmul1 = attn.matmul1
                matmul2 = attn.matmul2
                fc1 = child_module.mlp.fc1
                fc2 = child_module.mlp.fc2
                ###quantize norm1 
                print("start quant normqkv")
                alph_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
                matmul2.use_quantizer_A = False
                search_alph(alph_values, child_module, input_data, output_data, name, qkv, [qkv, matmul1, matmul2])
                matmul2.use_quantizer_A = True
                matmul2.quantizer_A.inited= False
                
                #quantize proj
                print("start quant normproj")
                alph_values = [-1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
                search_alph(alph_values, child_module, input_data, output_data, name, proj, [proj])
                set_quant_state(proj, input_quant=True, weight_quant=True)

                print("start quant normfc1")
                alph_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
                search_alph(alph_values, child_module, input_data, output_data, name, fc1, [fc1])

                print("start quant normfc2")
                alph_values = [-1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] if args.calib_batchsize >=256 else [-1]
                search_alph(alph_values, child_module, input_data, output_data, name, fc2, [fc2])
  
                set_quant_state(child_module, input_quant=True, weight_quant=True)
            
            elif isinstance(child_module, PatchMerging):
                input_data, _ = save_inp_oup_data(q_model, child_module, calib_dataset[0][0], store_inp=True, store_oup=False, bs=args.calib_batchsize)
                _, output_data = save_inp_oup_data(model, getattr(fp_module, name), calib_dataset[0][0], store_inp=False, store_oup=True, bs=args.calib_batchsize)
                reduction = child_module.reduction
                alph_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
                search_alph(alph_values, child_module, input_data, output_data, name, reduction, [reduction])
                set_quant_state(child_module, input_quant=True, weight_quant=True)

            elif getattr(fp_module, name, None) is not None:
                recon_model(child_module, getattr(fp_module, name), name) 

    set_quant_state(q_model, input_quant=False, weight_quant=False)
    recon_model(q_model, model)
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.inference_mode():
        for calib_data in tqdm(calib_dataset, desc="Calibration Progress", unit="batch"):
            calib_data = calib_data[0].to(device)
            _ = q_model(calib_data)
            break
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    # Validate the quantized model
    print("Validating ...")
    val_loss, val_prec1, val_prec5 = validate(
        args, val_loader, q_model, criterion, device
    )   

def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        target = target.to(device)
        data = data.to(device)
        target = target.to(device)

        with torch.inference_mode():
            output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
    val_end_time = time.time()
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
        top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser('UQ-ViT', parents=[get_args_parser()])
    args = parser.parse_args()
    main()