from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.openearthmap import OpenEarthMapDataset
from src.transforms import get_train_transform, get_val_transform


def create_train_dataloader(
    data_root,
    batch_size=4,
    image_size=512,
    num_workers=2,
    pin_memory=True,
):
    data_root = Path(data_root)

    train_dataset = OpenEarthMapDataset(
        image_dir=data_root / "images" / "train",
        mask_dir=data_root / "labels" / "train",
        transform=get_train_transform(image_size=image_size),
    )

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_val_dataloader(
    data_root,
    batch_size=4,
    image_size=512,
    num_workers=2,
    pin_memory=True,
):
    data_root = Path(data_root)

    val_dataset = OpenEarthMapDataset(
        image_dir=data_root / "images" / "val",
        mask_dir=data_root / "labels" / "val",
        transform=get_val_transform(image_size=image_size),
    )

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
