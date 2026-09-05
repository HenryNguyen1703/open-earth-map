import numpy as np
import torch

from src.transforms import (
    get_train_transform,
    get_val_transform,
)


def test_train_transform_returns_512_tensor():
    image = np.random.randint(
        0,
        256,
        size=(1000, 1000, 3),
        dtype=np.uint8,
    )

    mask = np.random.randint(
        0,
        9,
        size=(1000, 1000),
        dtype=np.uint8,
    )

    transform = get_train_transform(image_size=512)

    transformed = transform(
        image=image,
        mask=mask,
    )

    image_out = transformed["image"]
    mask_out = transformed["mask"]

    assert isinstance(image_out, torch.Tensor)
    assert isinstance(mask_out, torch.Tensor)

    assert image_out.shape == (3, 512, 512)
    assert mask_out.shape == (512, 512)


def test_val_transform_keeps_mask_class_ids():
    image = np.random.randint(
        0,
        256,
        size=(1000, 1000, 3),
        dtype=np.uint8,
    )

    mask = np.zeros(
        (1000, 1000),
        dtype=np.uint8,
    )

    mask[:, 500:] = 3

    transform = get_val_transform(image_size=512)

    transformed = transform(
        image=image,
        mask=mask,
    )

    mask_out = transformed["mask"]

    unique_values = set(
        torch.unique(mask_out).tolist()
    )

    assert unique_values.issubset({0, 3})