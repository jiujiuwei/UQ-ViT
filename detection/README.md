## UQ-ViT: Harmonizing Extreme Activations with Hardware-Friendly Uniform Quantization in Vision Transformers

Below are instructions for reproducing the detection results of UQ-ViT.
This repository is adopted from [*mmdetection*](https://github.com/open-mmlab/mmdetection) repo.

## Preliminaries

- Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```bash
pip install -U openmim
mim install mmcv-full
```

- Install MMDetection.

```bash
cd  detection
pip install -v -e .
```

- Download pre-trained models from [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) 

## Evaluation

- You can quantize and evaluate a single model using the following command:

```bash
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm [--w_bits] [--a_bits]
python tools/test.py ./configs/swin <DET_CHECKPOINT_FILE> --eval bbox segm [--w_bits] [--a_bits]
python tools/test.py ./configs/swin <DET_CHECKPOINT_FILE> --eval bbox segm 
Required arguments:
 <CONFIG_FILE> : Path to config. You can find it at ./configs/swin/
 <DET_CHECKPOINT_FILE> : Path to checkpoint of pre-trained models.

optional arguments:
--w_bit: Bit-precision of weights, default=4.
--a_bit: Bit-precision of activation, default=4.
```

- Example: Quantize *Mask R-CNN with Swin-T* at W4/A4 precision:

```bash
python tools/test.py /configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py /checkpoints/mask_rcnn_swin_tiny_patch4_window7.pth  --eval bbox segm
```

## Results

Below are the experimental results of our proposed RepQ-ViT that you should get on COCO dataset.

| Model                                     | Prec. | AP<sup>box</sup> / AP<sup>mask</sup> |
|:-----------------------------------------:|:-----:|:------------------------------------:|
| Mask RCNN w. Swin_T (46.0 / 41.6)         | W4/A4 | 39.0 / 37.3                         |
|                                           | W6/A6 | 45.6 / 41.4                         |
| Mask RCNN w. Swin_S (48.5 / 43.3)         | W4/A4 | 43.3 / 40.3                          |
|                                           | W6/A6 | 48.0 / 43.0                          |
| Cascade Mask RCNN w. Swin_T (50.4 / 43.7) | W4/A4 | 47.3 / 41.6                          |
|                                           | W6/A6 | 50.1 / 43.7                         |
| Cascade Mask RCNN w. Swin_S (51.9 / 45.0) | W4/A4 | 49.6 / 43.4                          |
|                                           | W6/A6 | 51.5 / 44.7                          |
