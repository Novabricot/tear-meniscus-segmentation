import os
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset


class TearMeniscusDataset(Dataset):
    def __init__(
        self,
        processed_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.split = split
        self.root = Path(processed_root)
        self.images_dir = self.root / split / "images"
        self.masks_dir = self.root / split / "masks"
        self.transform = transform

        if not self.images_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Processed dataset split not found: {self.images_dir} or {self.masks_dir}"
            )

        self.image_paths = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )

        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.masks_dir / image_path.name

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = image.astype("float32") / 255.0
            mask = mask.astype("float32") / 255.0
            image = np.transpose(image, (2, 0, 1))
            mask = np.expand_dims(mask, axis=0)

        mask = mask.float() if hasattr(mask, "float") else mask
        return {"image": image, "mask": mask}

    def set_transform(self, transform: Callable):
        self.transform = transform


def build_dataloader(
    processed_root: str,
    split: str,
    transform: Optional[Callable],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
):
    dataset = TearMeniscusDataset(processed_root=processed_root, split=split, transform=transform)
    return dataset, dataset, dataset


if __name__ == "__main__":
    from scripts.utils.augmentation import build_validation_transforms

    dataset = TearMeniscusDataset("data/processed", split="train", transform=build_validation_transforms(256,256))
    print(f"Loaded {len(dataset)} samples")
