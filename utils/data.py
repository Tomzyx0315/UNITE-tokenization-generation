"""
Data loading and dataset-profile helpers for training.
"""

from __future__ import annotations

import io
import os
import tarfile
from array import array
from pathlib import Path
from typing import Tuple

import numpy as np
import torch.distributed as dist
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.crop import CenterCropTransform


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXTENSIONS


class ImageNetTarDataset(Dataset):
    """Random-access reader for official ImageNet tar archives.

    Supports both the nested training archive:
      ILSVRC2012_img_train.tar -> n01440764.tar -> *.JPEG
    and a flat image tar such as the official validation archive. Images are
    read by byte offset from the tar, so the archive never needs to be extracted.
    """

    def __init__(self, tar_path: str | Path, transform=None, index_path: str | Path | None = None) -> None:
        self.tar_path = str(Path(tar_path).expanduser())
        self.transform = transform
        self.index_path = str(Path(index_path).expanduser()) if index_path is not None else None
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}
        self.offsets = array("Q")
        self.sizes = array("I")
        self.targets = array("H")
        self._file = None
        self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        if self.index_path is not None and Path(self.index_path).is_file():
            self._load_index(self.index_path)
            return

        self._build_index()
        if self.index_path is not None:
            self._save_index(self.index_path)

    def _tar_signature(self) -> tuple[int, int]:
        stat = Path(self.tar_path).stat()
        return stat.st_size, stat.st_mtime_ns

    def _load_index(self, index_path: str | Path) -> None:
        tar_size, tar_mtime_ns = self._tar_signature()
        with np.load(index_path, allow_pickle=False) as index:
            if "tar_size" in index and int(index["tar_size"]) != tar_size:
                raise RuntimeError(f"Tar index size mismatch: {index_path} does not match {self.tar_path}")
            if "tar_mtime_ns" in index and int(index["tar_mtime_ns"]) != tar_mtime_ns:
                raise RuntimeError(f"Tar index mtime mismatch: {index_path} does not match {self.tar_path}")

            self.classes = [str(item) for item in index["classes"].tolist()]
            self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
            self.offsets = array("Q", index["offsets"].astype(np.uint64).tolist())
            self.sizes = array("I", index["sizes"].astype(np.uint32).tolist())
            self.targets = array("H", index["targets"].astype(np.uint16).tolist())

        if len(self.offsets) == 0:
            raise RuntimeError(f"No images found in tar index: {index_path}")

    def _save_index(self, index_path: str | Path) -> None:
        index_path = Path(index_path).expanduser()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = index_path.with_name(f"{index_path.name}.tmp.{os.getpid()}")
        tar_size, tar_mtime_ns = self._tar_signature()
        try:
            with open(tmp_path, "wb") as handle:
                np.savez(
                    handle,
                    format_version=np.array(1, dtype=np.int32),
                    tar_size=np.array(tar_size, dtype=np.int64),
                    tar_mtime_ns=np.array(tar_mtime_ns, dtype=np.int64),
                    classes=np.asarray(self.classes),
                    offsets=np.asarray(self.offsets, dtype=np.uint64),
                    sizes=np.asarray(self.sizes, dtype=np.uint32),
                    targets=np.asarray(self.targets, dtype=np.uint16),
                )
            os.replace(tmp_path, index_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _build_index(self) -> None:
        with tarfile.open(self.tar_path, mode="r:") as outer_tar:
            members = [member for member in outer_tar.getmembers() if member.isfile()]
            class_tar_members = sorted(
                [member for member in members if member.name.lower().endswith(".tar")],
                key=lambda member: member.name,
            )
            if class_tar_members:
                self._build_nested_imagenet_index(outer_tar, class_tar_members)
            else:
                self._build_flat_image_index(members)

        if len(self.offsets) == 0:
            raise RuntimeError(f"No images found inside tar archive: {self.tar_path}")

    def _build_nested_imagenet_index(
        self,
        outer_tar: tarfile.TarFile,
        class_tar_members: list[tarfile.TarInfo],
    ) -> None:
        for class_idx, class_member in enumerate(class_tar_members):
            class_name = Path(class_member.name).stem
            self.classes.append(class_name)
            self.class_to_idx[class_name] = class_idx

            inner_file = outer_tar.extractfile(class_member)
            if inner_file is None:
                continue
            with inner_file:
                with tarfile.open(fileobj=inner_file, mode="r:") as inner_tar:
                    for image_member in inner_tar:
                        if image_member.isfile() and is_image_file(image_member.name):
                            self.offsets.append(class_member.offset_data + image_member.offset_data)
                            self.sizes.append(image_member.size)
                            self.targets.append(class_idx)

    def _build_flat_image_index(self, members: list[tarfile.TarInfo]) -> None:
        self.classes = ["unknown"]
        self.class_to_idx = {"unknown": 0}
        for member in sorted(members, key=lambda item: item.name):
            if member.isfile() and is_image_file(member.name):
                self.offsets.append(member.offset_data)
                self.sizes.append(member.size)
                self.targets.append(0)

    def __len__(self) -> int:
        return len(self.offsets)

    def _get_file(self):
        if self._file is None:
            self._file = open(self.tar_path, "rb")
        return self._file

    def __getitem__(self, index: int):
        file = self._get_file()
        file.seek(self.offsets[index])
        image_bytes = file.read(self.sizes[index])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[index])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        file = getattr(self, "_file", None)
        if file is not None:
            file.close()


def resolve_dataset_path(data_path: str | Path) -> Path:
    path = Path(data_path).expanduser()
    if path.is_dir():
        train_tar = path / "ILSVRC2012_img_train.tar"
        if train_tar.is_file():
            return train_tar
    return path


def get_transform_name(transform_type: int) -> str:
    transform_names = {
        0: "center_crop_arr+HFlip (JiT)",
        1: "Resize+RandomCrop",
        2: "RandomResizedCrop+HFlip",
    }
    return transform_names.get(transform_type, "unknown")


def prepare_dataloader(
    data_path,
    image_size,
    batch_size,
    num_workers,
    rank,
    world_size,
    transform_type: int = 0,
    rrc_scale_min: float = 0.8,
    rrc_scale_max: float = 1.0,
    data_index_path=None,
):
    if transform_type == 1:
        first_crop_size = 384 if image_size == 256 else int(image_size * 1.5)
        transform = transforms.Compose([
            transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
        ])
    elif transform_type == 2:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(rrc_scale_min, rrc_scale_max),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            CenterCropTransform(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    resolved_data_path = resolve_dataset_path(data_path)
    if resolved_data_path.is_file() and resolved_data_path.suffix.lower() == ".tar":
        if data_index_path is not None and dist.is_available() and dist.is_initialized():
            if rank == 0:
                dataset = ImageNetTarDataset(
                    resolved_data_path,
                    transform=transform,
                    index_path=data_index_path,
                )
            dist.barrier()
            if rank != 0:
                dataset = ImageNetTarDataset(
                    resolved_data_path,
                    transform=transform,
                    index_path=data_index_path,
                )
            dist.barrier()
        else:
            dataset = ImageNetTarDataset(
                resolved_data_path,
                transform=transform,
                index_path=data_index_path,
            )
    else:
        dataset = datasets.ImageFolder(str(resolved_data_path), transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, sampler
