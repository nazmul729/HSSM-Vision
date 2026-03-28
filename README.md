# HSSM-Vision (PyTorch)

This package contains a research-ready PyTorch implementation of **HSSM-Vision**:
- convolution-free patch tokenization via a fixed harmonic (DCT-style) basis
- four-directional 2D state-space mixing
- low-rank feed-forward layers
- classification backbone plus a minimal detection-ready scaffold
- ImageNet-style training script with AdamW, cosine decay, EMA, and MixUp

## Files
- `hssm_vision.py` — backbone, blocks, and detection scaffold
- `train_imagenet.py` — classification training script
- `README.md` — usage notes

## Important note
This code is a **training-ready research scaffold**. It is aligned with the methodology described in the paper draft, but the exact benchmark numbers discussed earlier are **not guaranteed** without substantial tuning, large-scale training, and careful reproduction settings.

## Quick start
```bash
python train_imagenet.py /path/to/imagenet \
  --model small \
  --epochs 300 \
  --batch-size 128 \
  --lr 1e-3 \
  --amp \
  --output outputs_hssm_small
```

Expected dataset layout:
```text
/path/to/imagenet/
  train/
    class_1/
    class_2/
    ...
  val/
    class_1/
    class_2/
    ...
```

## Programmatic usage
```python
import torch
from hssm_vision import hssm_vision_small, HSSMDetector

model = hssm_vision_small(num_classes=1000)
x = torch.randn(1, 3, 224, 224)
logits = model(x)
print(logits.shape)

detector = HSSMDetector(model, num_classes=80)
out = detector(x)
print([t.shape for t in out["cls_logits"]])
```

## Suggested tuning for stronger results
- increase image size to 256 or 384
- use stronger augmentation (RandAugment, repeated augmentation)
- distributed training and larger global batch size
- longer warmup and EMA tuning
- stage-specific state dimensions and scan fusion tuning
- stronger detection head and assignment/loss strategy for COCO
