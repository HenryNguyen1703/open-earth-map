import os

import cv2
import torch
from torch.utils.data import Dataset

class OpenEarthMapDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.file_names = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]

        image_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)

        # OpenCV đọc BGR
        image = cv2.imread(image_path)

        # Chuyển BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mask giữ nguyên class ID
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

        else: 
            # HWC -> CHW
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            image = image / 255.0

            mask = torch.from_numpy(mask).long()

        return image, mask