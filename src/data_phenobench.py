import os
import glob
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


@dataclass
class DataPaths:
    img_dir: str
    mask_dir: str


@dataclass(frozen=True)
class TaskConfig:
    task: str
    num_classes: int
    class_names: Tuple[str, ...]


TASK_CONFIGS = {
    "multiclass": TaskConfig(
        task="multiclass",
        num_classes=3,
        class_names=("soil", "crop", "weed"),
    ),
    "weed_binary": TaskConfig(
        task="weed_binary",
        num_classes=2,
        class_names=("non_weed", "weed"),
    ),
}


def get_task_config(task: str) -> TaskConfig:
    try:
        return TASK_CONFIGS[task]
    except KeyError as exc:
        valid = ", ".join(sorted(TASK_CONFIGS))
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {valid}") from exc


def _find_split_dirs(data_root: str, split: str) -> DataPaths:
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    # Common PhenoBench naming
    candidates_img = ["images", "image", "rgb", "imgs"]
    candidates_mask = ["semantics", "semantic", "labels", "masks"]

    img_dir = None
    mask_dir = None

    for c in candidates_img:
        p = os.path.join(split_dir, c)
        if os.path.isdir(p):
            img_dir = p
            break

    for c in candidates_mask:
        p = os.path.join(split_dir, c)
        if os.path.isdir(p):
            mask_dir = p
            break

    if img_dir is None or mask_dir is None:
        subdirs = [d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)]
        raise RuntimeError(
            f"Could not auto-detect image/mask dirs in: {split_dir}\n"
            f"Found subdirs: {subdirs}\n"
            f"Expected something like: {split_dir}/images and {split_dir}/semantics\n"
            f"Edit candidates in _find_split_dirs() if needed."
        )

    return DataPaths(img_dir=img_dir, mask_dir=mask_dir)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load_mask(path: str) -> np.ndarray:
    # masks can be stored as 8-bit or 16-bit png; keep integer ids
    m = np.array(Image.open(path))
    if m.ndim == 3:
        # sometimes masks are stored as RGB; convert to single channel if needed
        m = m[..., 0]
    return m.astype(np.int64)


def _map_labels(mask: np.ndarray, task: str = "multiclass") -> np.ndarray:
    """
    PhenoBench semantic segmentation task.

    Many releases encode:
      0=soil/background, 1=crop, 2=weed
    Some versions include partial labels:
      3=partial-crop, 4=partial-weed
    We merge partial -> main class.
    """
    m = mask.copy()

    # Merge partials if present
    m[m == 3] = 1
    m[m == 4] = 2

    if task == "weed_binary":
        # Weed is the positive class. Soil, crop, and merged partial-crop
        # become background/non-weed.
        return (m == 2).astype(np.int64)

    # Optional ignore index (if present in some label maps). If you see
    # strange values, print uniques in the train script and adjust here.
    return m


class PhenoBenchSemantics(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 512,
        augment: bool = False,
        task: str = "multiclass",
    ):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.task_config = get_task_config(task)
        self.task = self.task_config.task
        self.num_classes = self.task_config.num_classes
        self.class_names = self.task_config.class_names

        paths = _find_split_dirs(data_root, split)
        self.img_dir = paths.img_dir
        self.mask_dir = paths.mask_dir

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        imgs: List[str] = []
        for e in exts:
            imgs.extend(glob.glob(os.path.join(self.img_dir, e)))
        self.img_paths = sorted(imgs)

        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # Build mask paths by matching file stem
        self.mask_paths = []
        for ip in self.img_paths:
            name = _stem(ip)
            mp = os.path.join(self.mask_dir, name + ".png")
            if not os.path.isfile(mp):
                # Try any extension
                cands = glob.glob(os.path.join(self.mask_dir, name + ".*"))
                if len(cands) == 0:
                    raise RuntimeError(f"Missing mask for image {ip}. Expected {mp}")
                mp = cands[0]
            self.mask_paths.append(mp)

    def __len__(self) -> int:
        return len(self.img_paths)

    def _augment_pair(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Random vertical flip (optional; can be removed)
        if torch.rand(1).item() < 0.1:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask_np = _load_mask(self.mask_paths[idx])
        mask_np = _map_labels(mask_np, task=self.task)

        img_t = TF.to_tensor(img)  # [3,H,W] float in [0,1]
        mask_t = torch.from_numpy(mask_np).long()  # [H,W]

        # Resize image + mask to training size (fast demo)
        img_t = TF.resize(img_t, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask_t = TF.resize(mask_t.unsqueeze(0), [self.image_size, self.image_size],
                           interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        if self.augment and self.split == "train":
            img_t, mask_t = self._augment_pair(img_t, mask_t)

        # Normalize (ImageNet-style; ok for demo)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img_t, mask_t
