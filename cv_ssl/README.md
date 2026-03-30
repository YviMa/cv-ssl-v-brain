# Temporal Slowness in Central Vision Drives Semantic Object Learning

[![Paper](https://img.shields.io/badge/ICLR_2026-OpenReview-blue)](https://openreview.net/forum?id=tEezDE0vWt)
[![arXiv](https://img.shields.io/badge/arXiv-2602.04462-b31b1b.svg)](https://arxiv.org/abs/2602.04462)
[![Model Weights](https://img.shields.io/badge/Zenodo-Model%20Weights-024d81?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAACXBIWXMAAAsTAAALEwEAmpwYAAAE9GlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgNy4xLWMwMDAgNzkuZWRhMmIzZmFjLCAyMDIxLzExLzE3LTE3OjIzOjE5ICAgICAgICAiPiA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1sbnM6cGhvdG9zaG9wPSJodHRwOi8vbnMuYWRvYmUuY29tL3Bob3Rvc2hvcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgMjMuMSAoTWFjaW50b3NoKSIgeG1wOkNyZWF0ZURhdGU9IjIwMjMtMTAtMDRUMTQ6NDQ6MDIrMDI6MDAiIHhtcDpNb2RpZnlEYXRlPSIyMDIzLTEwLTA0VDE0OjU3OjQ2KzAyOjAwIiB4bXA6TWV0YWRhdGFEYXRlPSIyMDIzLTEwLTA0VDE0OjU3OjQ2KzAyOjAwIiBkYzpmb3JtYXQ9ImltYWdlL3BuZyIgcGhvdG9zaG9wOkNvbG9yTW9kZT0iMyIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDozZmQ0NmNlYy01YTc3LTRjMjMtYjZiOC1hY2IwNjJiYzliODgiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6M2ZkNDZjZWMtNWE3Ny00YzIzLWI2YjgtYWNiMDYyYmM5Yjg4IiB4bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ9InhtcC5kaWQ6M2ZkNDZjZWMtNWE3Ny00YzIzLWI2YjgtYWNiMDYyYmM5Yjg4Ij4gPHhtcE1NOkhpc3Rvcnk+IDxyZGY6U2VxPiA8cmRmOmxpIHN0RXZ0OmFjdGlvbj0iY3JlYXRlZCIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDozZmQ0NmNlYy01YTc3LTRjMjMtYjZiOC1hY2IwNjJiYzliODgiIHN0RXZ0OndoZW49IjIwMjMtMTAtMDRUMTQ6NDQ6MDIrMDI6MDAiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCAyMy4xIChNYWNpbnRvc2gpIi8+IDwvcmRmOlNlcT4gPC94bXBNTTpIaXN0b3J5PiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI/Pmx8nAYAAAPbSURBVGiB7ZpfiFVVFMZ/d/6E1oRplhU6hgoaIgpJgSCIEJb5J0MLfDJKJwTfIhQf8iF8iOghEgpBSiGqMUwxfBEV8kUKAxERhlJxGMUc/4Bmztzp82Hfcc7sWfves/fcaze5Hxzu7G+ftfb67lp7n73PnYIkHgY0/dcBVAsNIfWGhpB6Q0NIvaElwE8FxlTBv4A/gAGjrwDMBJ4FJpXal4FLwNn4kST/6lR10S1pdsb/WEnbJPVUsNkm6QkjPvPKNpoldVRZxCB+K42xXlJvhN0tSZsVKaRN0t5qRe6hW9KBUdgfVAUhBQ3fay0Ffo6uzweDA8DKUKcvBOAD4G1gAvYkDeFvoAi8GGFzC/fFXSi124HXgceNewV8CHxqeqqUssjr24hy+Uj2ZB4naWvA5oqkFkkF366aIrZEiFiaw98SSUXPrk/Sx5KerpWQlyNEdET4XWvYX5C0uhZCxkq6mlPExQT/JzwfdyXtkCux+/dVY4vyA/CkwfcbnD1Ry8O3eQSYBzyWJUcrZBOwzOC/Bj4z+OMJY/QY3GRgYpYYjZAFwOcGXwTeAVqNvjsJ41iPgDZgSpZIFdICHAz0rSp9zjD6LiWMZWVkDDDNDygFPwLjDX4XQwKLRv9bQBfQnHOcAWCOwTcDz+B2zII0IRuBFQb/O/Bupm29Z/oyYbwQHs02YktrFrDD4O9SZh9UIzThMnK/EYNDAX4dcDExoBQUyIiAOCE/Ac8b/E7gu+SQ0nAdOJcl8s6R97BL5xywISKAk8AN3EMtFQLO4Krj30Eyj5DJuG/dwpLIIJaRtgRXRJ7SOhbg1+GW0hjf43KMl4RKGdkFTDf43cA3FWytJ3JqSS0COnBLbhH4B/gedygbgPJC1uC2Gj7O47JRCd0GNx84lcM2i3bgqMf1AvvJPKtCpdWO29VaeAX7YefDeje1JYedj0+8tnBL/elhcQTOAF2B88T7EeeIuQEfb0b4eMGwvyZ3SmzL3msZfxEIYG9EAIPXLwFf03PYTpB03rPrl3RU0jxJTeWEvBYY+HiCCCS1Srpt+OuT9EYZu1cl/WXY9UraJC8bkoZN9rXAV4E67Qf2YL+mKYeb2CfFVmAf8Gfp81fc6jMXdwyYbdgIt+ncB9we2TukansgG/WCIxrK8ogsZletwgiV9YNTwPLS31aGhwk5A/TVMJj9uLeYsdgJvIRVTll4KVpeo7K4LOmp0hiLJe2Wm/Dl0ClplV9Coct69zsHWIjL1mj/m6CAy3InbuudxXO4TecU3CJSKH3rPcBh3A9E+QcyhPwv8dD8htgQUm9oCKk3NITUG+4BHP2mwwGKgTwAAAAASUVORK5CYII=&logoColor=white&style=flat)](https://zenodo.org/records/19191064)


This is the official code for the paper **"Temporal Slowness in Central Vision Drives Semantic Object Learning"**.

![Main Figure](assets/main_figure.png)


---

## Installation

```bash
git clone https://github.com/your-org/central-vision-ssl.git
cd central-vision-ssl
pip install -r requirements.txt
```

---


## Pretrained Model Weights

Download all pretrained checkpoints from Zenodo: 

https://zenodo.org/records/19191064

### Load a pretrained model

```python
import torch
from omegaconf import OmegaConf
from solo.methods import METHODS

ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
cfg = OmegaConf.create(ckpt["args"])

model = METHODS[cfg["method"]](cfg)
model.load_state_dict(ckpt["state_dict"], strict=True)
```

---
## Dataset

### Ego4D
Download the full Ego4D dataset from https://ego4d-data.org/

### Gaze Annotations
Follow the instructions at https://github.com/Aubret/GLC/blob/main/GENERATION.md in the GLC repository.

---

## Training

Before training, set your data paths in `scripts/pretrain/ego4d/paths/default.yaml`:

```yaml
data_train_path: /path/to/ego4d/h5
checkpoint_dir: /path/to/save/checkpoints
knn_clb_train_path: /path/to/imagenet100/train
knn_clb_val_path: /path/to/imagenet100/val
```

**ResNet-50:**
```bash
python main_pretrain.py --config-path scripts/pretrain/ego4d/ --config-name mocov3_resnet50.yaml
```

**ViT-Base:**
```bash
python main_pretrain.py --config-path scripts/pretrain/ego4d/ --config-name mocov3_vit.yaml
```

### Key parameters

| Parameter | Description |
|---|---|
| `data.dataset_kwargs.time_window` | Number of frames between the two views sampled for contrastive learning. Larger values enforce slower temporal invariance. |
| `data.dataset_kwargs.gaze_size` | Size of the gaze-centered crop (in pixels). Set to `540` to use the full frame instead of a gaze crop. |
| `data.dataset_kwargs.center_crop` | Set to `True` to crop around the frame center instead of the gaze location. |

These can be overridden on the command line, e.g.:

```bash
python main_pretrain.py --config-path scripts/pretrain/ego4d/ --config-name mocov3_resnet50.yaml \
  data.dataset_kwargs.time_window=20 \
  data.dataset_kwargs.gaze_size=224 \
  data.dataset_kwargs.center_crop=False
```

---

## Linear Probe

Set your data paths in `scripts/linear/paths/default.yaml`, then run:

```bash
python main_linear.py --config-path scripts/linear/ --config-name mocov3_resnet50.yaml
```

---

## Semantic Alignment Analysis

The `model_semantic_alignment/` directory contains a pipeline for measuring how well SSL model representations align with object co-occurrence structure (COCO, Visual Genome, ADE20K).

See [model_semantic_alignment/README.md](model_semantic_alignment/README.md) for full usage instructions.

---

## Citation

```bibtex
@inproceedings{
schaumloffel2026temporal,
title={Temporal Slowness in Central Vision Drives Semantic Object Learning},
author={Timothy Schauml{\"o}ffel and Arthur Aubret and Gemma Roig and Jochen Triesch},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=tEezDE0vWt}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
