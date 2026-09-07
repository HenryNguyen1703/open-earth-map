import cv2
import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler


def _write_sample(image_dir, mask_dir, name):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)

    cv2.imwrite(str(image_dir / name), image)
    cv2.imwrite(str(mask_dir / name), mask)


def _make_split(tmp_path, split, count=4):
    image_dir = tmp_path / "images" / split
    mask_dir = tmp_path / "labels" / split

    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for index in range(count):
        _write_sample(image_dir, mask_dir, f"sample_{index}.tif")


def test_create_train_dataloader_shuffles_and_returns_expected_shapes(tmp_path):
    from src.dataloaders import create_train_dataloader

    _make_split(tmp_path, "train")

    loader = create_train_dataloader(
        data_root=tmp_path,
        batch_size=2,
        image_size=32,
        num_workers=0,
        pin_memory=False,
    )

    images, masks = next(iter(loader))

    assert isinstance(loader.sampler, RandomSampler)
    assert images.shape == (2, 3, 32, 32)
    assert masks.shape == (2, 32, 32)


def test_create_val_dataloader_does_not_shuffle_and_returns_expected_shapes(tmp_path):
    from src.dataloaders import create_val_dataloader

    _make_split(tmp_path, "val")

    loader = create_val_dataloader(
        data_root=tmp_path,
        batch_size=2,
        image_size=32,
        num_workers=0,
        pin_memory=False,
    )

    images, masks = next(iter(loader))

    assert isinstance(loader.sampler, SequentialSampler)
    assert images.shape == (2, 3, 32, 32)
    assert masks.shape == (2, 32, 32)
