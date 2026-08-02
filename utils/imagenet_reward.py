"""ImageNet class reward helpers for class-conditional RL fine-tuning."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


RewardMode = Literal["logprob", "prob", "margin", "accuracy"]


class DINOv2ImageNetReward:
    """Frozen DINOv2 ImageNet classifier reward.

    DMDR uses DINOv2 with a linear ImageNet head as a class-conditional reward
    model. For AWM we use the detached scalar score, typically log p(y | image),
    rather than backpropagating through the reward model.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        model_name: str = "dinov2_vitl14_lc",
        dtype: torch.dtype = torch.float32,
        mode: RewardMode = "logprob",
    ) -> None:
        if mode not in {"logprob", "prob", "margin", "accuracy"}:
            raise ValueError(f"Unsupported reward mode: {mode}")

        self.device = device
        self.dtype = dtype
        self.mode = mode
        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                trust_repo=True,
            )
        except TypeError:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(device=device, dtype=dtype)
        self.model.eval().requires_grad_(False)

        mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device, dtype=torch.float32)
        std = torch.tensor(IMAGENET_DEFAULT_STD, device=device, dtype=torch.float32)
        self.mean = mean.view(1, 3, 1, 1)
        self.std = std.view(1, 3, 1, 1)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = images.detach().to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
        images = F.interpolate(
            images,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.mean) / self.std
        return images.to(dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(self.device, dtype=torch.long)
        logits = self.model(self._preprocess(images)).float()
        target_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)

        if self.mode == "logprob":
            return F.log_softmax(logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
        if self.mode == "prob":
            return F.softmax(logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
        if self.mode == "accuracy":
            return (logits.argmax(dim=-1) == labels).to(dtype=torch.float32)

        masked = logits.clone()
        masked.scatter_(1, labels.view(-1, 1), float("-inf"))
        return target_logits - masked.max(dim=-1).values
