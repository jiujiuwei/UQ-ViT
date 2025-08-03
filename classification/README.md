## UQ-ViT: Harmonizing Extreme Activations with Hardware-Friendly Uniform Quantization in Vision Transformers

Below are instructions for reproducing the classification results of UQ-ViT.

## Evaluation

- You can quantize and evaluate a single model using the following command:

```bash
python test.py [--model] [--dataset] [--w_bit] [--a_bit]

optional arguments:
--model: Model architecture, the choises can be: 
    vit_small, vit_base, deit_tiny, deit_small, deit_base, swin_tiny, swin_small.
--dataset: Path to ImageNet dataset.
--w_bit: Bit-precision of weights, default=4.
--a_bit: Bit-precision of activation, default=4.
```

- Example: Quantize *DeiT-S* at W4/A4 precision:

```bash
python test.py --model deit_small --dataset <YOUR_DATA_DIR>
```

## Results

Below are the experimental results of our proposed UQ-ViT that you should get on ImageNet dataset.

| Model          | Prec. | Top-1(%) | Prec. | Top-1(%) |
|:--------------:|:-----:|:--------:|:-----:|:--------:|
| ViT-S (81.39)  | W4/A4 | 68.34    | W6/A6 | 80.71    |
| ViT-B (84.54)  | W4/A4 | 72.07    | W6/A6 | 83.81    |
| DeiT-T (72.21) | W4/A4 | 59.71    | W6/A6 | 71.17    |
| DeiT-S (79.85) | W4/A4 | 72.13    | W6/A6 | 79.07    |
| DeiT-B (81.80) | W4/A4 | 76.59    | W6/A6 | 81.45    |
| Swin-S (83.23) | W4/A4 | 79.93    | W6/A6 | 82.82    |
| Swin-B (85.27) | W4/A4 | 81.96    | W6/A6 | 84.99    |
