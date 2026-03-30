from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from solo.data.custom.base import H5ClassificationDataset


class ImgNetDataset(H5ClassificationDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        split: str = "train",
        subset: Optional[str] = None,
    ):
        super().__init__(
            root,
            transform,
            split,
            driver="core" if split == "train" and subset is not None else None,
        )
        if subset == "1pct" and split == "train":
            subset_df = pd.read_csv(
                "solo/data/dataset_subset/imagenet_1percent.txt", names=["filename"]
            )
            self.mapper = (
                self.mapper.query("filename.isin(@subset_df.filename)")
                .copy()
                .reset_index(drop=True)
            )
            print("Using IN 1%")
        elif subset == "10pct" and split == "train":
            subset_df = pd.read_csv(
                "solo/data/dataset_subset/imagenet_10percent.txt", names=["filename"]
            )
            self.mapper = (
                self.mapper.query("filename.isin(@subset_df.filename)")
                .copy()
                .reset_index(drop=True)
            )
            print("Using IN 10%")
        elif subset == "imgnet100":
            with open(self.root / "imagenet100_classes.txt") as f:
                imgnet100_classes = sorted(f.readline().strip().split())
            imgnet100_class_wn_2_class_index = {
                class_wn: class_index
                for class_index, class_wn in enumerate(imgnet100_classes)
            }

            self.mapper = self.mapper.query(
                "wn_name in @imgnet100_classes"
            ).reset_index(drop=True)
            self.mapper["target"] = self.mapper["wn_name"].apply(
                lambda x: imgnet100_class_wn_2_class_index[x]
            )

        self.n_classes = self.mapper["target"].nunique()
        self.target_2_class_name = (
            self.mapper[["target", "class_name"]]
            .drop_duplicates()
            .set_index("target")["class_name"]
            .to_dict()
        )


class ImageNetOODDataset(Dataset):
    VERSION = [
        "colour",
        "contrast",
        "cue-conflict",
        "edge",
        "eidolonI",
        "eidolonII",
        "eidolonIII",
        "false-colour",
        "high-pass",
        "low-pass",
        "phase-scrambling",
        "power-equalisation",
        "rotation",
        "silhouette",
        "sketch",
        "stylized",
        "uniform-noise",
    ]

    def __init__(
        self,
        path: str,
        version: str,
        return_classname: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.version = version
        self.transform = transform
        self.return_classname = return_classname
        self.h5_file = h5py.File(path, "r", driver="core")

        self.available_versions = list(self.h5_file.keys())
        if not self.version in self.available_versions:
            raise ValueError(
                f"Version {self.version} not available, choose one of {self.available_versions}"
            )

    def __len__(self) -> int:
        return self.h5_file.get(self.version).get("targets").shape[0]

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Union[str, np.ndarray]]:
        version = self.h5_file.get(self.version)
        image = Image.fromarray(version.get("images")[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_classname:
            targets = version.get("classnames")[idx]
        else:
            targets = version.get("targets")[idx]

        return image, targets


class ImageNetS(ImageFolder):
    def __init__(self, root, transform=None, mode: str = "full"):
        super().__init__(root, transform)
        mask_p = root + "-segmentation/"
        self.masks = [
            Path(mask_p + "/".join(x[0].split("/")[-2:])).with_suffix(".png")
            for x in self.samples
        ]
        self.mode = mode
        if self.mode not in ["full", "fg", "bg", "bg_rec"]:
            raise ValueError(
                f"Mode {self.mode} not available, choose one of ['full', 'fg', 'bg', 'bg_rec']"
            )

    def __getitem__(self, index: int):
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.mode == "fg":
            mask = np.array(Image.open(self.masks[index]))[..., 0] > 0
            sample = Image.fromarray(np.array(sample) * mask[..., None])
        elif self.mode == "bg":
            mask = 1 - (np.array(Image.open(self.masks[index]))[..., 0] > 0).astype(
                np.uint8
            )
            sample = Image.fromarray(np.array(sample) * mask[..., None])
        elif self.mode == "bg_rec":
            mask = np.array(Image.open(self.masks[index]))[..., 0]
            bbox = cv2.boundingRect(mask)
            mask = np.ones_like(mask)
            mask[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]] = 0
            sample = Image.fromarray(np.array(sample) * mask[..., None])

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
