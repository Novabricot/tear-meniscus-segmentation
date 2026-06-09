from typing import Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _scale_probability(value: float, intensity: float) -> float:
    return min(1.0, max(0.0, float(value) * float(intensity)))


def _elastic_transform(cfg: Dict, intensity: float):
    # Albumentations v2 removed alpha_affine. Keep only stable arguments.
    return A.ElasticTransform(
        alpha=cfg.get("alpha", 1.0),
        sigma=cfg.get("sigma", 5.0),
        p=_scale_probability(cfg.get("p", 0.3), intensity),
    )


def build_train_transforms(
    config: Dict,
    height: int,
    width: int,
    intensity: float = 1.0,
):
    aug_config = config.get("augmentation", {})
    aug = []

    # Photometric augmentations: image only, no shape change.
    if aug_config.get("brightness_contrast", {}).get("p", 0) > 0:
        cfg = aug_config["brightness_contrast"]
        aug.append(
            A.RandomBrightnessContrast(
                brightness_limit=cfg.get("brightness_limit", 0.2),
                contrast_limit=cfg.get("contrast_limit", 0.2),
                p=_scale_probability(cfg.get("p", 0.3), intensity),
            )
        )

    if aug_config.get("gamma", {}).get("p", 0) > 0:
        cfg = aug_config["gamma"]
        aug.append(
            A.RandomGamma(
                gamma_limit=tuple(cfg.get("gamma_limit", [80, 120])),
                p=_scale_probability(cfg.get("p", 0.2), intensity),
            )
        )

    if aug_config.get("clahe", {}).get("p", 0) > 0:
        cfg = aug_config["clahe"]
        aug.append(A.CLAHE(p=_scale_probability(cfg.get("p", 0.2), intensity)))

    if aug_config.get("gaussian_noise", {}).get("p", 0) > 0:
        cfg = aug_config["gaussian_noise"]
        # Albumentations v1 uses var_limit; v2 accepts std_range. This remains OK on v1.
        aug.append(
            A.GaussNoise(
                var_limit=tuple(cfg.get("var_limit", [10, 50])),
                p=_scale_probability(cfg.get("p", 0.2), intensity),
            )
        )

    if aug_config.get("iso_noise", {}).get("p", 0) > 0:
        cfg = aug_config["iso_noise"]
        aug.append(A.ISONoise(p=_scale_probability(cfg.get("p", 0.1), intensity)))

    # Spatial augmentations: can change orientation/shape.
    if aug_config.get("horizontal_flip", {}).get("p", 0) > 0:
        cfg = aug_config["horizontal_flip"]
        aug.append(A.HorizontalFlip(p=_scale_probability(cfg.get("p", 0.5), intensity)))

    # For this medical segmentation task, vertical flip / rotate90 can create unrealistic anatomy
    # and also caused shape swaps [H,W] -> [W,H]. Leave disabled by default in YAML.
    if aug_config.get("vertical_flip", {}).get("p", 0) > 0:
        cfg = aug_config["vertical_flip"]
        aug.append(A.VerticalFlip(p=_scale_probability(cfg.get("p", 0.0), intensity)))

    if aug_config.get("random_rotate_90", {}).get("p", 0) > 0:
        cfg = aug_config["random_rotate_90"]
        aug.append(A.RandomRotate90(p=_scale_probability(cfg.get("p", 0.0), intensity)))

    if aug_config.get("elastic_transform", {}).get("p", 0) > 0:
        aug.append(_elastic_transform(aug_config["elastic_transform"], intensity))

    if aug_config.get("grid_distortion", {}).get("p", 0) > 0:
        cfg = aug_config["grid_distortion"]
        aug.append(A.GridDistortion(p=_scale_probability(cfg.get("p", 0.2), intensity)))

    # IMPORTANT: final resize must be after every spatial transform, otherwise RandomRotate90
    # can produce mixed shapes inside the same batch.
    aug.extend(
        [
            A.Resize(int(height), int(width)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(aug)


def build_validation_transforms(height: int, width: int):
    return A.Compose(
        [
            A.Resize(int(height), int(width)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def progressive_intensity(config: Dict, epoch: int, steps: int) -> float:
    progressive = config.get("augmentation", {}).get("progressive_augmentation", {})
    if not progressive.get("enabled", False) or steps <= 0:
        return 1.0

    start = float(progressive.get("start_intensity", 0.3))
    end = float(progressive.get("end_intensity", 1.0))
    intensity = start + (end - start) * min(1.0, epoch / max(1, steps - 1))
    return float(max(0.0, min(1.0, intensity)))
