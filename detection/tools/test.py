import copy
import argparse
import copy
import os
import warnings
import mmcv
import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from quant import *


def search_alph(alph_values, child_module, input_data, output_data, name, search_module, modules, input_mask = None, H = None, W = None):
    best_score = 1e+10
    best_alph = 0
    for module in modules:
        set_quant_state(module, input_quant=True, weight_quant=True)
    with torch.no_grad():
        for alph in alph_values:

            search_module.alph = alph
            for module in modules:
                set_initquant_state(module, False)
            if input_mask is not None:
                output = child_module(input_data.clone(), input_mask.clone())
            elif H is not None and W is not None:
                output = child_module(input_data.clone(), H, W)
            else:
                output = child_module(input_data.clone())
            score = lp_loss(output, output_data, p=2, reduction='all')
            if score < best_score:
                best_score = score
                best_alph = alph
        search_module.alph = best_alph
        for module in modules:
            set_initquant_state(module, False)
        if input_mask is not None:
            output = child_module(input_data.clone(), input_mask.clone())
        elif H is not None and W is not None:
            output = child_module(input_data.clone(), H, W)
        else:
            output = child_module(input_data.clone())  
        score = lp_loss(output, output_data, p=2, reduction='all')        
        if score == best_score:
            print("name: ", name, "best_alph: ", best_alph, "best_score: ", best_score,)  
        else:
            print("error")
            print("name: ", name, "best_alph: ", best_alph, "best_score: ", best_score)   
        for module in modules:
            set_initquant_state(module, True)
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

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default="/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py", help='test config file path')
    parser.add_argument('--checkpoint', default="/checkpoints/mask_rcnn_swin_tiny_patch4_window7.pth",help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['bbox', 'segm'],
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--w_bits', default=4,
                        type=int, help='bit-precision of weights')
    parser.add_argument('--a_bits', default=4,
                        type=int, help='bit-precision of activation')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    cfg.data.test.samples_per_gpu = 2
    print("cfg.data.test.samples_per_gpu: ", cfg.data.test.samples_per_gpu)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES


    # build the quantized model
    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model(copy.deepcopy(model), input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.cuda()
    q_model.eval()

    if not distributed:
        q_model = MMDataParallel(q_model, device_ids=[0])
        model = MMDataParallel(model, device_ids=[0])
        dataset = data_loader.dataset
        from mmdet.models.backbones.swin_transformer import (
            PatchEmbed, PatchMerging, SwinTransformerBlock)
        
        with torch.inference_mode():
            for i, calib_data in enumerate(data_loader):
                # result = q_model(return_loss=False, rescale=True, **calib_data)  #(1, 3, 800, 1216)
                break
        
        def save_inp_oup_data(model, module, calib_data, store_inp=False, store_oup=False, keep_gpu: bool = True):
            device = next(model.parameters()).device
            data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
            handle = module.register_forward_hook(data_saver)
            cached = [[], []]
            with torch.inference_mode():
                    try:
                        _ = model(return_loss=False, rescale=True, **calib_data)
                    except StopForwardException:
                        pass
                    if store_inp:
                        if keep_gpu:
                            cached[0].append(data_saver.input_store)
                        else:
                            cached[0].append(data_saver.input_store[0].detach().cpu())
                    if store_oup:
                        if keep_gpu:
                            cached[1].append(data_saver.output_store.detach())
                        else:
                            cached[1].append(data_saver.output_store.detach().cpu())
            if store_oup:
                cached[1] = torch.cat([x for x in cached[1]])
            handle.remove()
            torch.cuda.empty_cache()
            return cached
        
        @torch.inference_mode()
        def recon_model(module: nn.Module, fp_module: nn.Module, name_module = None):

            for name, child_module in module.named_children():
                if isinstance(child_module, (PatchEmbed, )):
                    input_data, _ = save_inp_oup_data(q_model, child_module, calib_data, store_inp=True, store_oup=False)
                    input_data = input_data[0][0]
                    set_quant_state(child_module, input_quant=True, weight_quant=True)
                    output_data = child_module(input_data)

                elif isinstance(child_module, (SwinTransformerBlock)):

                    input_data, _ = save_inp_oup_data(q_model, child_module, calib_data, store_inp=True, store_oup=False)
                    input_data, input_mask = input_data[0][0], input_data[0][1]
                    _, output_data = save_inp_oup_data(model, getattr(fp_module, name), calib_data, store_inp=False, store_oup=True)
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
                    search_alph(alph_values, child_module, input_data, output_data, name, qkv, [qkv, matmul1, matmul2], input_mask = input_mask)
                    matmul2.use_quantizer_A = True

                    ###quantize proj
                    print("start quant normproj")
                    alph_values = [-1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                    search_alph(alph_values, child_module, input_data, output_data, name, proj, [proj], input_mask = input_mask)
                    set_quant_state(proj, input_quant=True, weight_quant=True)

                    print("start quant normfc1")
                    alph_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
                    search_alph(alph_values, child_module, input_data, output_data, name, fc1, [fc1], input_mask = input_mask)

                    print("start quant normfc2")
                    alph_values = [-1]
                    search_alph(alph_values, child_module, input_data, output_data, name, fc2, [fc2], input_mask = input_mask)
    
                    set_quant_state(child_module, input_quant=True, weight_quant=True)
                
                elif isinstance(child_module, PatchMerging):
                    input_data, _ = save_inp_oup_data(q_model, child_module, calib_data, store_inp=True, store_oup=False)
                    input_data, H, W = input_data[0][0], input_data[0][1], input_data[0][2]
                    
                    _, output_data = save_inp_oup_data(model, getattr(fp_module, name), calib_data, store_inp=False, store_oup=True)
                    reduction = child_module.reduction
                    alph_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
                    print("start quant reduction")
                    search_alph(alph_values, child_module, input_data, output_data, name, reduction, [reduction], H = H, W = W)
                    set_quant_state(child_module, input_quant=True, weight_quant=True)

                elif getattr(fp_module, name, None) is not None:
                    recon_model(child_module, getattr(fp_module, name), name) 

        set_quant_state(q_model, input_quant=False, weight_quant=False)
        # set_quant_state(q_model, input_quant=True, weight_quant=True)
        recon_model(q_model, model)

        set_quant_state(q_model, input_quant=True, weight_quant=True)

        with torch.inference_mode():
            result = q_model(return_loss=False, rescale=True, **calib_data)

        outputs = single_gpu_test(q_model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
