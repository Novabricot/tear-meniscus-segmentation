from typing import Dict, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _scale_probability(value: float, intensity: float) -> float:
    return min(1.0, max(0.0, value * intensity))


def build_train_transforms(
    config: Dict,
    height: int,
    width: int,
    intensity: float = 1.0,
):
    aug = []
    aug.append(A.Resize(height, width))

    aug_config = config.get("augmentation", {})

    if aug_config.get("brightness_contrast", {}).get("p", 0) > 0:
        aug.append(
            A.RandomBrightnessContrast(
                brightness_limit=aug_config["brightness_contrast"].get("brightness_limit", 0.2),
                contrast_limit=aug_config["brightness_contrast"].get("contrast_limit", 0.2),
                p=_scale_probability(aug_config["brightness_contrast"].get("p", 0.3), intensity),
            )
        )

    if aug_config.get("gamma", {}).get("p", 0) > 0:
        aug.append(
            A.RandomGamma(
                gamma_limit=tuple(aug_config["gamma"].get("gamma_limit", [80, 120])),
                p=_scale_probability(aug_config["gamma"].get("p", 0.2), intensity),
            )
        )

    if aug_config.get("clahe", {}).get("p", 0) > 0:
        aug.append(
            A.CLAHE(p=_scale_probability(aug_config["clahe"].get("p", 0.2), intensity))
        )

    if aug_config.get("elastic_transform", {}).get("p", 0) > 0:
        aug.append(
            A.ElasticTransform(
                alpha=aug_config["elastic_transform"].get("alpha", 1.0),
                sigma=aug_config["elastic_transform"].get("sigma", 5.0),
                alpha_affine=aug_config["elastic_transform"].get("alpha_affine", 5.0),
                p=_scale_probability(aug_config["elastic_transform"].get("p", 0.3), intensity),
            )
        )

    if aug_config.get("grid_distortion", {}).get("p", 0) > 0:
        aug.append(
            A.GridDistortion(p=_scale_probability(aug_config["grid_distortion"].get("p", 0.2), intensity))
        )

    if aug_config.get("gaussian_noise", {}).get("p", 0) > 0:
        aug.append(
            A.GaussNoise(
                var_limit=tuple(aug_config["gaussian_noise"].get("var_limit", [10, 50])),
                p=_scale_probability(aug_config["gaussian_noise"].get("p", 0.2), intensity),
            )
        )

    if aug_config.get("iso_noise", {}).get("p", 0) > 0:
        aug.append(
            A.ISONoise(p=_scale_probability(aug_config["iso_noise"].get("p", 0.1), intensity))
        )

    if aug_config.get("horizontal_flip", {}).get("p", 0) > 0:
        aug.append(
            A.HorizontalFlip(p=_scale_probability(aug_config["horizontal_flip"].get("p", 0.5), intensity))
        )

    if aug_config.get("vertical_flip", {}).get("p", 0) > 0:
        aug.append(
            A.VerticalFlip(p=_scale_probability(aug_config["vertical_flip"].get("p", 0.2), intensity))
        )

    if aug_config.get("random_rotate_90", {}).get("p", 0) > 0:
        aug.append(
            A.RandomRotate90(p=_scale_probability(aug_config["random_rotate_90"].get("p", 0.2), intensity))
        )

    aug.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return A.Compose(aug)


def build_validation_transforms(height: int, width: int):
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def progressive_intensity(config: Dict, epoch: int, steps: int) -> float:
    progressive = config.get("augmentation", {}).get("progressive_augmentation", {})
    if not progressive.get("enabled", False) or steps <= 0:
        return 1.0

    start = progressive.get("start_intensity", 0.3)
    end = progressive.get("end_intensity", 1.0)
    intensity = start + (end - start) * min(1.0, epoch / max(1, steps - 1))
    return float(max(0.0, min(1.0, intensity)))
