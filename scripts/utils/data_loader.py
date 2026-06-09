from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.masks_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for image {image_path.name}: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Albumentations expects masks as HxW arrays.
        # Force binary mask before transforms to avoid strange gray levels.
        mask = (mask > 127).astype("float32")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = image.astype("float32") / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).float()

        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image).float()
        else:
            image = image.float()

        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask).float()
        else:
            mask = mask.float()

        # DataLoader needs all masks to be [1, H, W], not [H, W].
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1)

        # Safety: keep masks binary after interpolation/augmentations.
        mask = (mask > 0.5).float()

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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, loader


if __name__ == "__main__":
    from scripts.utils.augmentation import build_validation_transforms

    dataset = TearMeniscusDataset(
        "data/processed",
        split="train",
        transform=build_validation_transforms(512, 512),
    )
    sample = dataset[0]
    print(f"Loaded {len(dataset)} samples")
    print("image", sample["image"].shape, sample["image"].dtype)
    print("mask", sample["mask"].shape, sample["mask"].dtype)
