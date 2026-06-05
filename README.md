# CNN From Scratch & Food-101 Classifier

A deep learning project in two parts: implementing core CNN layers from scratch in PyTorch, then training a ResNet-18 classifier on the Food-101 dataset.

## Part 1 — CNN Layers from Scratch

Implemented the following operations using only basic PyTorch tensor ops — no `for` loops, no `torch.nn` layer modules:

- 2D Convolution (`Conv2d`)
- 2D Max Pooling (`MaxPool2d`)
- Linear layer
- 2D Batch Normalisation (`BatchNorm2d`)

Unit tests for each layer are in [tests/](tests/).

## Part 2 — Food-101 Image Classifier

Trained a modified ResNet-18 from scratch on the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (101 food categories, 101,000 images).

**Architecture changes:**
- LeakyReLU (negative slope 0.1) in residual blocks instead of ReLU, to avoid dying ReLU
- BatchNorm throughout for regularisation instead of dropout

**Training setup:**
- AdamW optimiser (lr=0.001, weight_decay=1e-4)
- Cosine annealing learning rate scheduler
- Early stopping to prevent overfitting

**Data augmentation (train only):**
- RandomResizedCrop(256)
- RandomHorizontalFlip
- RandomRotation(±15°)
- ColorJitter (brightness, contrast)

Also includes out-of-distribution evaluation on a custom hot dog dataset, and feature map visualisation across network layers.

## Setup

```bash
pip install -r requirements.txt
```

Model weights (`.pt` files) are not included. Run the training cells in `dl_cw_1.ipynb` to reproduce them.

The Food-101 dataset is not included. Download instructions are in the notebook.
