import albumentations as A
import cv2

from albumentations.pytorch import ToTensorV2

def get_train_transform(image_size=512):
    return A.Compose([
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        ),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.Normalize(
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ])

def get_val_transform(image_size=512):
    return A.Compose([
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        ),

        A.Normalize(
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ])