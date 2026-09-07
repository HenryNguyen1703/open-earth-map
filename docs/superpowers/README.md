# Superpowers Project Context — OpenEarthMap Semantic Segmentation

> Purpose: this file stores the working context and development rules for the OpenEarthMap research project so a future ChatGPT/Superpowers session can read it and continue from the correct technical state without reconstructing the whole conversation.

## 1. Project identity

- Repository: `HenryNguyen1703/open-earth-map`
- Local path on macOS: `/Users/nguyengochieu/Documents/OpenEarthMap/exercise-oem/open-earth-map`
- Main stack: Python, PyTorch, Albumentations, OpenCV, pytest
- Training environment: Kaggle
- Task: semantic segmentation on OpenEarthMap
- GitHub SSH is already configured and normal push/pull works.

## 2. Dataset

Kaggle dataset path:

```text
/kaggle/input/datasets/dyiyacao/openearthmap
```

Expected structure:

```text
openearthmap/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Known counts:

```text
Train images: 2149
Train labels: 2149
Val images:   538
Val labels:   538
```

Image and label filenames match directly, for example:

```text
images/train/aachen_10.tif
labels/train/aachen_10.tif
```

## 3. Dataset loader

Main file:

```text
src/datasets/openearthmap.py
```

`OpenEarthMapDataset` behavior:

- Reads images with OpenCV.
- Converts BGR to RGB.
- Reads masks using `cv2.IMREAD_UNCHANGED`.
- Accepts a transform from outside the dataset class.
- Returns:
  - `image`: `torch.float32`
  - `mask`: `torch.int64`
- Without Albumentations transform, image is converted from `[H, W, 3]` to `[3, H, W]` and divided by `255.0`.

## 4. Transforms

Main file:

```text
src/transforms.py
```

Functions:

```python
get_train_transform(image_size=512)
get_val_transform(image_size=512)
```

Train transform pipeline:

```text
Resize 512x512
→ HorizontalFlip
→ VerticalFlip
→ RandomRotate90
→ Normalize
→ ToTensorV2
```

Validation transform pipeline:

```text
Resize 512x512
→ Normalize
→ ToTensorV2
```

Important segmentation rule:

```python
mask_interpolation=cv2.INTER_NEAREST
```

Never resize a class-ID mask with bilinear/bicubic interpolation because interpolation can create invalid class values between existing IDs.

Transform tests live in:

```text
tests/test_transforms.py
```

They should verify at least:

```text
image -> [3, 512, 512]
mask  -> [512, 512]
```

and verify that resizing the mask does not introduce new class IDs.

## 5. DataLoader target behavior

Expected loaders:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
```

Expected batch shapes:

```text
Images: [4, 3, 512, 512]
Masks:  [4, 512, 512]
```

## 6. Model direction

Build U-Net manually with PyTorch for learning/research purposes.

Configuration:

```text
in_channels  = 3
num_classes  = 9
base_channels = 32
```

Planned construction order:

```text
ConvBlock
→ EncoderBlock
→ Bottleneck
→ DecoderBlock
→ UNet
```

Do not implement the whole U-Net at once.

Target high-level architecture:

```text
Input
[B, 3, 512, 512]
        ↓
Encoder
32 → 64 → 128 → 256 → 512
        ↓
Bottleneck
        ↓
Decoder + skip connections
        ↓
1x1 Conv
        ↓
Output logits
[B, 9, 512, 512]
```

Do **not** put `softmax` inside the model because training will use:

```python
nn.CrossEntropyLoss()
```

Expected loss shapes:

```text
logits: [B, 9, H, W]
target: [B, H, W]
```

The target mask contains integer class IDs and must be `torch.int64` / `LongTensor` for `CrossEntropyLoss`.

## 7. TDD / Superpowers workflow

For every model component use this exact development loop:

```text
RED
→ write the smallest meaningful failing test
→ run pytest and observe the expected failure

GREEN
→ write the minimum production code required to satisfy the test
→ run pytest and confirm it passes

REFACTOR
→ improve naming/readability only if useful
→ run pytest again
```

Rules:

1. Test first.
2. Do not write future components before their tests exist.
3. Never claim a test passes without running it.
4. Verify the expected tensor shapes explicitly.
5. Keep code simple and readable for a learner.
6. Explain the reason for each layer and tensor transformation.
7. Do not add unnecessary abstractions.
8. Do not create extra documentation files unless they genuinely help the project.
9. After each component passes, stop and explain the next component before implementing it.

## 8. ConvBlock design

Planned `ConvBlock`:

```text
Conv2d
→ BatchNorm2d
→ ReLU
→ Conv2d
→ BatchNorm2d
→ ReLU
```

Each convolution:

```python
kernel_size=3
padding=1
bias=False
```

With `stride=1`, a `3x3` convolution with `padding=1` preserves spatial size.

Example shape:

```text
[B, 3, 64, 64]
→ ConvBlock(3, 32)
→ [B, 32, 64, 64]
```

The block changes feature channels but preserves height and width.

Minimal intended test:

```python
import torch

from src.models.blocks import ConvBlock


def test_conv_block_preserves_spatial_size_and_changes_channels():
    block = ConvBlock(
        in_channels=3,
        out_channels=32,
    )

    x = torch.randn(2, 3, 64, 64)

    output = block(x)

    assert output.shape == (2, 32, 64, 64)
```

## 9. EncoderBlock design

Do this only after `ConvBlock` is GREEN.

Proposed encoder behavior:

```text
Input x
[B, 3, 64, 64]
        ↓
ConvBlock(3, 32)
        ↓
features / skip
[B, 32, 64, 64]
        ↓
MaxPool2d(kernel_size=2, stride=2)
        ↓
pooled
[B, 32, 32, 32]
```

Recommended return value:

```python
return features, pooled
```

Why return two tensors:

- `features` is saved for the future U-Net skip connection.
- `pooled` is passed down to the next encoder stage.

Minimal intended shape assertions:

```python
assert features.shape == (2, 32, 64, 64)
assert pooled.shape == (2, 32, 32, 32)
```

## 10. Concepts that must be explained while mentoring

### Channels

For a tensor `[B, C, H, W]`:

- `B`: batch size
- `C`: channels / feature maps
- `H`: height
- `W`: width

RGB starts with `C=3`. Inside the network, channels such as `32`, `64`, `128` are learned feature maps, not RGB colors.

### Spatial size

Spatial size means `H x W`.

Example:

```text
64x64 → 32x32
```

means spatial resolution was reduced, usually by pooling or strided convolution.

### MaxPool

Typical U-Net encoder pooling:

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

For even spatial sizes it halves height and width:

```text
64x64 → 32x32
```

Channels stay unchanged.

### ConvTranspose2d

Used later in the decoder to increase spatial resolution.

Typical conceptual effect:

```text
[B, C, 32, 32]
→ upsample
→ [B, C_out, 64, 64]
```

Explain exact channel arguments when implementing `DecoderBlock`; do not introduce it prematurely.

### Skip connection

U-Net keeps high-resolution encoder features and transfers them directly to the matching decoder level.

This helps recover precise object boundaries and spatial detail that can be lost during downsampling.

### Concatenation

The decoder combines an upsampled tensor with the matching encoder skip tensor along the **channel dimension**:

```python
torch.cat([up, skip], dim=1)
```

Example:

```text
up   = [B, 256, 64, 64]
skip = [B, 256, 64, 64]

concat along channels
→ [B, 512, 64, 64]
```

Height and width must match before concatenation.

### Why U-Net fits semantic segmentation

Semantic segmentation requires a class prediction for every pixel.

U-Net is suitable because:

- Encoder learns increasingly abstract semantic features.
- Downsampling increases the effective receptive field.
- Decoder restores spatial resolution.
- Skip connections preserve fine local details and boundaries.
- Final `1x1 Conv` maps feature channels to one logit channel per semantic class.

## 11. Testing commands

Run all tests:

```bash
python -m pytest -q
```

Run only U-Net tests:

```bash
python -m pytest tests/test_unet.py -q
```

Run only the ConvBlock test:

```bash
python -m pytest \
  tests/test_unet.py::test_conv_block_preserves_spatial_size_and_changes_channels \
  -q
```

For TDD, always observe RED before writing production code for a new component.

## 12. Last verified local repository state

Important: the conversation described `ConvBlock` as already written, but a direct local repository inspection on **2026-09-06** showed that the checked-out repo did not yet contain:

```text
src/models/blocks.py
tests/test_unet.py
```

The locally observed tree at that moment contained the dataset/transforms work, and `git status` was clean.

Therefore, future sessions must trust the **filesystem + git state** over conversational assumptions.

Before continuing model work, run:

```bash
git status --short
find src tests -maxdepth 3 -type f | sort
```

Then inspect whether `tests/test_unet.py` and `src/models/blocks.py` exist.

## 13. Immediate next step

Unless the filesystem has changed since the last verification, continue here:

```text
1. Create tests/test_unet.py with the ConvBlock test.
2. Run the test and observe RED.
3. Create src/models/blocks.py with the minimum ConvBlock implementation.
4. Run the ConvBlock test and confirm GREEN.
5. Only then write the EncoderBlock test.
6. Observe RED.
7. Implement EncoderBlock using ConvBlock + MaxPool2d.
8. Confirm both ConvBlock and EncoderBlock tests are GREEN.
```

Do not jump directly to Bottleneck, DecoderBlock, or full UNet.

## 14. Mentor style for future sessions

When guiding this project:

- Proceed one small step at a time.
- Explain tensor shapes at every important boundary.
- Explain **why** a test or layer exists, not just what to type.
- Give commands the user can copy and run.
- Do not silently modify multiple unrelated files.
- Stop after the current TDD unit passes before moving on.
- Prefer readable beginner-friendly PyTorch over clever abstractions.
- When local state and chat context disagree, inspect the local files and use them as source of truth.

---

### Quick resume prompt for a future session

Read this file first:

```text
/Users/nguyengochieu/Documents/OpenEarthMap/exercise-oem/open-earth-map/docs/superpowers/README.md
```

Then verify the repository state and continue from the **Immediate next step** section using Superpowers/TDD.
