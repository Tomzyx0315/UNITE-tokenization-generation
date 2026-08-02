#!/usr/bin/env python
"""AWM fine-tuning entry point for pretrained UNITE.

This intentionally treats UNITE as a pretrained latent denoiser plus decoder:
roll out images from class labels, score them with a frozen reward model, and
apply advantage-weighted flow matching on the generated latents. Optionally,
the original UNITE reconstruction path is trained on real images in the same
optimizer step.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from engines.eval import evaluate_fid
from models.unite import UNITE
from utils.data import prepare_dataloader
from utils.distributed import cleanup_distributed, is_main_process, setup_distributed
from utils.ema import update_ema
from utils.imagenet_reward import DINOv2ImageNetReward
from utils.logging import create_logger


logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AWM fine-tune pretrained UNITE.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Pretrained UNITE checkpoint.")
    parser.add_argument("--data-path", type=str, default=None, help="ImageNet train directory for reconstruction.")
    parser.add_argument("--results-dir", type=str, default="results_awm")
    parser.add_argument("--experiment-name", type=str, default="unite-awm")
    return parser.parse_args()


def _normalize_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("module.")
        new_key = new_key.replace("._orig_mod.", ".")
        new_key = new_key.removeprefix("_orig_mod.")
        normalized[new_key] = value
    return normalized


def load_unite_checkpoint(policy: UNITE, ckpt_path: str, init_from: str = "ema") -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and init_from in checkpoint:
        state = checkpoint[init_from]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    msg = policy.load_state_dict(_normalize_checkpoint_keys(state), strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match UNITE architecture: "
            f"missing={msg.missing_keys}, unexpected={msg.unexpected_keys}"
        )


def freeze_non_policy_parts(policy: UNITE, *, freeze_decoder: bool, freeze_patch_embed: bool) -> None:
    if freeze_decoder:
        policy.decoder.requires_grad_(False)
        policy.up_sample_decoder.requires_grad_(False)
    if freeze_patch_embed:
        policy.patch_embed.requires_grad_(False)


def sample_grouped_labels(
    *,
    num_groups: int,
    samples_per_group: int,
    num_classes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_labels = torch.randint(num_classes, (num_groups,), device=device)
    labels = group_labels.repeat_interleave(samples_per_group)
    return labels, group_labels


def group_advantages(
    rewards: torch.Tensor,
    *,
    num_groups: int,
    samples_per_group: int,
    eps: float,
    clip: float | None,
    max_abs: float | None,
) -> torch.Tensor:
    grouped = rewards.detach().float().view(num_groups, samples_per_group)
    if samples_per_group > 1:
        adv = (grouped - grouped.mean(dim=1, keepdim=True)) / (
            grouped.std(dim=1, keepdim=True, unbiased=False) + eps
        )
    else:
        adv = (grouped - grouped.mean()) / (grouped.std(unbiased=False) + eps)
    adv = adv.view(-1)
    if clip is not None and clip > 0:
        adv = adv.clamp(-clip, clip)
    if max_abs is not None and clip is not None and clip > 0:
        adv = adv / clip * max_abs
    return adv


class UNITEAWMLoss(nn.Module):
    def __init__(
        self,
        policy: UNITE,
        reference: UNITE,
        kl_ema_reference: UNITE,
        awm_cfg: Dict[str, Any],
        reconstruction_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.policy = policy
        self.reference = reference.eval()
        self.reference.requires_grad_(False)
        self.kl_ema_reference = kl_ema_reference.eval()
        self.kl_ema_reference.requires_grad_(False)

        self.train_timesteps = int(awm_cfg.get("train_timesteps", 4))
        self.beta_kl = float(awm_cfg.get("beta_kl", 0.0))
        self.ema_beta = float(awm_cfg.get("ema_beta", 0.0))
        self.clip_range = awm_cfg.get("clip_range", None)
        self.clip_range = None if self.clip_range is None else float(self.clip_range)
        self.logprob_scale = float(awm_cfg.get("logprob_scale", 1.0))
        self.checkpoint_blocks = bool(awm_cfg.get("gradient_checkpointing", False))
        self.generation_weight = float(awm_cfg.get("generation_weight", getattr(policy, "gen_loss_weight", 1.0)))
        self.reconstruction_weight = float(reconstruction_cfg.get("weight", 1.0))

    def _predict_x0(self, model: UNITE, z_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_embed = model.latent_tokens[:, : z_t.shape[1]].expand(z_t.shape[0], -1, -1)
        force_keep = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)
        pred = model.encoder(
            z_t,
            t=t,
            y=labels,
            pos_embed=pos_embed,
            checkpoint_blocks=self.checkpoint_blocks and model.training,
            force_drop_ids_y_embedder=force_keep,
        )
        return model.encoder_layer_norm(pred)

    def _flow_logprob(
        self,
        model: UNITE,
        z_clean: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_view = t[:, None, None]
        z_t = t_view * z_clean + (1.0 - t_view) * noise
        x0_pred = self._predict_x0(model, z_t, t, labels)
        denom = (1.0 - t_view).clamp_min(model.transport.train_eps)
        v_pred = (x0_pred - z_t) / denom
        v_target = (z_clean - z_t) / denom
        mse = (v_pred - v_target).pow(2).mean(dim=(1, 2))
        return -self.logprob_scale * mse, v_pred

    def _awm_forward(
        self,
        z_clean: torch.Tensor,
        labels: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.reference.eval()
        self.kl_ema_reference.eval()

        batch = z_clean.shape[0]
        z_base = z_clean.detach()
        z_rep = z_base[:, None].expand(batch, self.train_timesteps, *z_base.shape[1:])
        z_rep = z_rep.reshape(batch * self.train_timesteps, *z_base.shape[1:])
        labels_rep = labels[:, None].expand(batch, self.train_timesteps).reshape(-1)
        adv_rep = advantages[:, None].expand(batch, self.train_timesteps).reshape(-1)

        t = self.policy.transport.sample(
            z_rep,
            timestep_shift=self.policy.timestep_shift_alpha,
        )[0]
        noise = torch.randn_like(z_rep)

        logprob, v_pred = self._flow_logprob(self.policy, z_rep, labels_rep, t, noise)
        old_logprob = logprob.detach()
        ratio = torch.exp(logprob - old_logprob)

        unclipped = -adv_rep * ratio
        if self.clip_range is not None and self.clip_range > 0:
            clipped_ratio = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
            policy_loss = torch.maximum(unclipped, -adv_rep * clipped_ratio).mean()
        else:
            policy_loss = unclipped.mean()

        if self.beta_kl > 0:
            with torch.no_grad():
                _, v_ref = self._flow_logprob(self.reference, z_rep, labels_rep, t, noise)
            kl_loss = (v_pred - v_ref).pow(2).mean(dim=(1, 2)).mean()
        else:
            kl_loss = torch.zeros((), device=z_clean.device, dtype=z_clean.dtype)

        if self.ema_beta > 0:
            with torch.no_grad():
                _, v_ema = self._flow_logprob(self.kl_ema_reference, z_rep, labels_rep, t, noise)
            ema_kl_loss = (v_pred - v_ema).pow(2).mean(dim=(1, 2)).mean()
        else:
            ema_kl_loss = torch.zeros((), device=z_clean.device, dtype=z_clean.dtype)

        loss = policy_loss + self.beta_kl * kl_loss + self.ema_beta * ema_kl_loss
        metrics = {
            "loss/policy": policy_loss.detach(),
            "loss/kl_base": kl_loss.detach(),
            "loss/kl_ema": ema_kl_loss.detach(),
            "loss/awm_total": loss.detach(),
            "awm/logprob": logprob.detach().mean(),
            "awm/ratio": ratio.detach().mean(),
            "awm/adv_abs": adv_rep.detach().abs().mean(),
        }
        return loss, metrics

    def _reconstruction_forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        recon_loss, recon_metrics = self.policy.forward_tokenizer(images)
        metrics = {
            "loss/recon_total": recon_metrics["recon_total"].detach(),
            "loss/recon_l1": recon_metrics["rec_loss"].detach(),
            "loss/recon_lpips": recon_metrics["lpips_loss"].detach(),
        }
        return recon_loss, metrics

    def forward(
        self,
        z_clean: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        recon_images: torch.Tensor | None = None,
        awm_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = self.policy.latent_tokens.device
        total_loss = torch.zeros((), device=device)
        metrics: Dict[str, torch.Tensor] = {}

        if z_clean is not None:
            if labels is None or advantages is None:
                raise ValueError("labels and advantages are required when z_clean is provided.")
            awm_loss, awm_metrics = self._awm_forward(z_clean, labels, advantages)
            weighted_awm = self.generation_weight * awm_loss_weight * awm_loss
            total_loss = total_loss + weighted_awm
            metrics.update(awm_metrics)
            metrics["loss/awm_weighted"] = weighted_awm.detach()

        if recon_images is not None and self.reconstruction_weight > 0:
            recon_loss, recon_metrics = self._reconstruction_forward(recon_images)
            weighted_recon = self.reconstruction_weight * recon_loss_weight * recon_loss
            total_loss = total_loss + weighted_recon
            metrics.update(recon_metrics)
            metrics["loss/recon_weighted"] = weighted_recon.detach()

        metrics["loss/total"] = total_loss.detach()
        return total_loss, metrics


@torch.no_grad()
def update_ema_policy(ema_policy: UNITE, current_policy: UNITE, decay: float) -> None:
    update_ema(ema_policy, current_policy, decay)


def get_kl_ema_decay(awm_cfg: Dict[str, Any], step: int) -> float:
    max_decay = float(awm_cfg.get("kl_ema_decay", 0.3))
    decay_type = str(awm_cfg.get("kl_ema_decay_type", "linear"))
    if decay_type == "constant":
        return max_decay
    if decay_type == "linear":
        return min(max_decay, 0.001 * step)
    raise ValueError(f"Unknown kl_ema_decay_type: {decay_type}")


def save_awm_checkpoint(
    path: str,
    *,
    step: int,
    awm_model: nn.Module,
    ema_policy: UNITE,
    kl_ema_policy: UNITE,
    optimizer: torch.optim.Optimizer,
    cfg_scale: float,
) -> None:
    raw = awm_model.module if hasattr(awm_model, "module") else awm_model
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": raw.policy.state_dict(),
            "ema": ema_policy.state_dict(),
            "kl_ema": kl_ema_policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "current_best_cfg_scale": cfg_scale,
        },
        path,
    )


def build_fixed_grid(
    policy: UNITE,
    sample_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    class_ids = [int(x) for x in sample_cfg.get("grid_classes", [])]
    if not class_ids:
        raise ValueError("sampling.grid_classes must be non-empty when grid_interval > 0.")
    if min(class_ids) < 0 or max(class_ids) >= policy.num_classes:
        raise ValueError(
            f"sampling.grid_classes must be in [0, {policy.num_classes - 1}], got {class_ids}."
        )

    samples_per_class = int(sample_cfg.get("grid_samples_per_class", 2))
    if samples_per_class <= 0:
        raise ValueError("sampling.grid_samples_per_class must be positive.")

    labels = torch.tensor(class_ids, dtype=torch.long).repeat_interleave(samples_per_class)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(sample_cfg.get("grid_seed", 12345)))
    noise = torch.randn(
        labels.shape[0],
        policy.num_latent_tokens,
        policy.diffusion_input_dim,
        generator=generator,
    )
    return labels.to(device), noise.to(device), samples_per_class


@torch.no_grad()
def save_sample_grid(
    policy: UNITE,
    *,
    sample_dir: str,
    step: int,
    labels: torch.Tensor,
    noise: torch.Tensor,
    nrow: int,
    device: torch.device,
    device_type: str,
    autocast_kwargs: Dict[str, Any],
    cfg_scale: float,
    cfg_interval: Tuple[float, float],
    cfg_norm_order: str,
) -> str:
    os.makedirs(sample_dir, exist_ok=True)
    policy.eval()
    with autocast(device_type, **autocast_kwargs):
        images = policy.diffusion_generate(
            device=device,
            num_visuals=labels.shape[0],
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            cfg_norm_order=cfg_norm_order,
            y_given=labels,
            noise=noise,
        )

    path = os.path.join(sample_dir, f"step_{step:07d}.png")
    save_image(images.clamp(0.0, 1.0).float().cpu(), path, nrow=nrow)
    return path


@torch.no_grad()
def rollout_policy_batch(
    policy: UNITE,
    reward_model: DINOv2ImageNetReward,
    *,
    labels: torch.Tensor,
    reward_scale: float,
    rollout_micro_batch_size: int,
    device: torch.device,
    device_type: str,
    autocast_kwargs: Dict[str, Any],
    cfg_scale: float,
    cfg_interval: Tuple[float, float],
    cfg_norm_order: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z_chunks, reward_chunks = [], []
    batch = labels.shape[0]
    rollout_micro_batch_size = batch if rollout_micro_batch_size <= 0 else rollout_micro_batch_size

    for start in range(0, batch, rollout_micro_batch_size):
        end = min(start + rollout_micro_batch_size, batch)
        y = labels[start:end]
        with autocast(device_type, **autocast_kwargs):
            images, _labels_np, z_clean = policy.diffusion_generate(
                device=device,
                num_visuals=y.shape[0],
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_norm_order=cfg_norm_order,
                y_given=y,
                return_z=True,
            )
        rewards = reward_model(images, y) * reward_scale
        z_chunks.append(z_clean.detach())
        reward_chunks.append(rewards.detach())
        del images, z_clean, rewards

    return torch.cat(z_chunks, dim=0), torch.cat(reward_chunks, dim=0)


def split_loss_indices(num_items: int, num_chunks: int, device: torch.device) -> list[torch.Tensor]:
    if num_items <= 0:
        raise ValueError("Cannot split an empty rollout batch.")
    num_chunks = min(max(int(num_chunks), 1), num_items)
    return list(torch.arange(num_items, device=device).chunk(num_chunks))


def split_indices_by_micro_batch_size(
    num_items: int,
    micro_batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    if num_items <= 0:
        raise ValueError("Cannot split an empty batch.")
    if micro_batch_size <= 0:
        micro_batch_size = num_items
    indices = torch.arange(num_items, device=device)
    return [indices[start:start + micro_batch_size] for start in range(0, num_items, micro_batch_size)]


class ReconstructionBatcher:
    def __init__(self, loader: Any, sampler: Any, device: torch.device) -> None:
        self.loader = loader
        self.sampler = sampler
        self.device = device
        self.epoch = 0
        self.iterator = iter(loader)

    def next_images(self) -> torch.Tensor:
        try:
            images, _labels = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            images, _labels = next(self.iterator)
        return images.to(self.device, non_blocking=True)


def metric_accum_weight(key: str, awm_weight: float, recon_weight: float) -> float:
    if key in {"loss/total", "loss/awm_weighted", "loss/recon_weighted"}:
        return 1.0
    if key.startswith("loss/recon"):
        return recon_weight
    return awm_weight


def _as_float_list(value: Any, fallback: float) -> list[float]:
    if value is None:
        value = [fallback]
    elif isinstance(value, (int, float)):
        value = [value]
    return [float(x) for x in value]


def _as_interval_list(value: Any, fallback: Tuple[float, float]) -> list[Tuple[float, float]]:
    if value is None:
        value = [fallback]
    elif (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(x, (int, float)) for x in value)
    ):
        value = [value]

    intervals = []
    for interval in value:
        if len(interval) != 2:
            raise ValueError(f"Eval CFG interval must have length 2, got {interval}.")
        intervals.append((float(interval[0]), float(interval[1])))
    return intervals


def _as_str_list(value: Any, fallback: str) -> list[str]:
    if value is None:
        value = [fallback]
    elif isinstance(value, str):
        value = [value]
    return [str(x) for x in value]


def _contains_tensor_state_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(torch.is_tensor(item) for item in value.values())


def validate_inception_weights(path: str) -> None:
    try:
        try:
            weights = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            weights = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load INCEPTION_WEIGHTS with torch.load: {path}") from exc

    valid = _contains_tensor_state_dict(weights)
    if not valid and isinstance(weights, dict):
        valid = any(_contains_tensor_state_dict(item) for item in weights.values())
    if not valid:
        kind = type(weights).__name__
        keys = list(weights.keys())[:10] if isinstance(weights, dict) else None
        raise RuntimeError(
            "INCEPTION_WEIGHTS must be a torch checkpoint containing a tensor state_dict; "
            f"got type={kind}, keys={keys} from {path}"
        )


def validate_checkpoint_eval_config(eval_cfg: Dict[str, Any]) -> None:
    if not bool(eval_cfg.get("enabled", False)):
        return
    required = ("INCEPTION_WEIGHTS", "IN256_FID_STATS")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Checkpoint FID evaluation is enabled, but these environment variables are missing: "
            f"{', '.join(missing)}. Set them or disable eval.enabled."
        )
    for name in required:
        path = os.environ[name]
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Checkpoint FID evaluation is enabled, but {name} does not point to a file: {path}"
            )

    validate_inception_weights(os.environ["INCEPTION_WEIGHTS"])

    stats_path = os.environ["IN256_FID_STATS"]
    try:
        with np.load(stats_path) as stats:
            keys = set(stats.files)
    except Exception as exc:
        raise RuntimeError(f"Failed to read IN256_FID_STATS as an .npz file: {stats_path}") from exc
    if not {"mu", "sigma"}.issubset(keys):
        raise RuntimeError(
            f"IN256_FID_STATS must contain 'mu' and 'sigma' arrays, got keys={sorted(keys)}"
        )


def run_checkpoint_eval(
    *,
    ema_policy: UNITE,
    reward_model: DINOv2ImageNetReward,
    eval_cfg: Dict[str, Any],
    sample_cfg: Dict[str, Any],
    device: torch.device,
    global_step: int,
    eval_dir: str,
    log: logging.Logger,
    autocast_kwargs: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not bool(eval_cfg.get("enabled", False)):
        return None

    cfg_scales = _as_float_list(eval_cfg.get("cfg_scales"), float(sample_cfg.get("cfg_scale", 1.0)))
    cfg_intervals = _as_interval_list(
        eval_cfg.get("cfg_intervals"),
        tuple(float(x) for x in sample_cfg.get("cfg_interval", (0.1, 1.0))),
    )
    cfg_norm_orders = _as_str_list(
        eval_cfg.get("cfg_norm_orders"),
        str(sample_cfg.get("cfg_norm_order", "norm_first")),
    )
    num_images = int(eval_cfg.get("num_images", 50000))
    num_classes = int(eval_cfg.get("num_classes", ema_policy.num_classes))
    image_size = int(eval_cfg.get("image_size", 256))
    batch_size = int(eval_cfg.get("batch_size", 50))
    offload_reward = bool(eval_cfg.get("offload_reward_model", True))

    if is_main_process():
        log.info(
            f"[Eval] Starting checkpoint eval at step {global_step}: "
            f"num_images={num_images}, batch_size={batch_size}, cfg_scales={cfg_scales}, "
            f"cfg_intervals={cfg_intervals}, cfg_norm_orders={cfg_norm_orders}"
        )

    if offload_reward and device.type == "cuda":
        reward_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()

    eval_start = time.time()
    try:
        ema_policy.eval()
        best_cfg, fid, is_score = evaluate_fid(
            ema_policy,
            device=device,
            global_step=global_step,
            image_size=image_size,
            cfg_scales=cfg_scales,
            cfg_intervals=cfg_intervals,
            cfg_norm_orders=cfg_norm_orders,
            num_images=num_images,
            num_classes=num_classes,
            batch_size=batch_size,
            log_to_wandb=False,
            log_fid_best=False,
            current_best_cfg_scale=cfg_scales[0],
            return_metrics=True,
            autocast_kwargs=autocast_kwargs,
        )
    finally:
        if dist.is_initialized():
            dist.barrier()
        if offload_reward and device.type == "cuda":
            reward_model.to(device)
            torch.cuda.empty_cache()

    metrics = {
        "step": int(global_step),
        "eval/fid50k": float(fid),
        "eval/is50k": float(is_score),
        "eval/best_cfg_scale": float(best_cfg),
        "eval/num_images": num_images,
        "eval/batch_size": batch_size,
        "eval/sec": time.time() - eval_start,
    }
    if is_main_process():
        os.makedirs(eval_dir, exist_ok=True)
        metrics_path = os.path.join(eval_dir, "metrics.jsonl")
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        log.info(
            f"[Eval] step={global_step}, FID-50K={fid:.4f}, IS-50K={is_score:.4f}, "
            f"best_cfg={best_cfg:.2f}, metrics={metrics_path}"
        )
    return metrics


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()

    with open(args.config, "r") as f:
        full_cfg = yaml.safe_load(f)

    training_cfg = dict(full_cfg.get("training", {}))
    gen_tok_cfg = dict(full_cfg.get("gen_tok", {}))
    awm_cfg = dict(full_cfg.get("awm", {}))
    reward_cfg = dict(full_cfg.get("reward", {}))
    sample_cfg = dict(full_cfg.get("sampling", {}))
    eval_cfg = dict(full_cfg.get("eval", {}))
    reconstruction_cfg = dict(full_cfg.get("reconstruction", {}))
    validate_checkpoint_eval_config(eval_cfg)

    seed = int(training_cfg.get("global_seed", 0)) * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if rank == 0:
        experiment_dir = os.path.join(args.results_dir, args.experiment_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        sample_dir = os.path.join(experiment_dir, "samples")
        eval_dir = os.path.join(experiment_dir, "eval")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        log = create_logger(experiment_dir)
        log.info(f"Experiment directory created at {experiment_dir}")
    else:
        checkpoint_dir = os.path.join(args.results_dir, args.experiment_name, "checkpoints")
        eval_dir = os.path.join(args.results_dir, args.experiment_name, "eval")
        sample_dir = None
        log = create_logger(None)

    policy = UNITE(gen_tok_cfg, num_classes=int(training_cfg.get("num_classes", 1000))).to(device)
    load_unite_checkpoint(policy, args.ckpt, init_from=str(training_cfg.get("init_from", "ema")))
    freeze_non_policy_parts(
        policy,
        freeze_decoder=bool(training_cfg.get("freeze_decoder", True)),
        freeze_patch_embed=bool(training_cfg.get("freeze_patch_embed", True)),
    )

    reference = deepcopy(policy).to(device).eval()
    reference.requires_grad_(False)
    ema_policy = deepcopy(policy).to(device).eval()
    ema_policy.requires_grad_(False)
    kl_ema_policy = deepcopy(policy).to(device).eval()
    kl_ema_policy.requires_grad_(False)

    awm_module = UNITEAWMLoss(policy, reference, kl_ema_policy, awm_cfg, reconstruction_cfg).to(device)
    if dist.is_initialized():
        ddp_model = DDP(
            awm_module,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    else:
        ddp_model = awm_module

    params = [p for p in policy.parameters() if p.requires_grad]
    opt_cfg = dict(training_cfg.get("optimizer", {}))
    optimizer = torch.optim.AdamW(
        params,
        lr=float(opt_cfg.get("lr", 1e-6)),
        betas=tuple(float(x) for x in opt_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )

    precision = str(training_cfg.get("precision", "bf16"))
    if precision == "bf16":
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    elif precision == "fp16":
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    else:
        autocast_kwargs = dict(enabled=False)

    reward_dtype = torch.float16 if precision == "fp16" else torch.float32
    reward_kwargs = dict(
        model_name=str(reward_cfg.get("model_name", "dinov2_vitl14_lc")),
        dtype=reward_dtype,
        mode=str(reward_cfg.get("mode", "logprob")),
    )
    if dist.is_initialized():
        if rank == 0:
            reward_model = DINOv2ImageNetReward(device, **reward_kwargs)
        dist.barrier()
        if rank != 0:
            reward_model = DINOv2ImageNetReward(device, **reward_kwargs)
        dist.barrier()
    else:
        reward_model = DINOv2ImageNetReward(device, **reward_kwargs)

    reconstruction_enabled = bool(reconstruction_cfg.get("enabled", False))
    reconstruction_batcher = None
    reconstruction_batch_size = int(reconstruction_cfg.get("batch_size", 64))
    reconstruction_micro_batch_size = int(
        reconstruction_cfg.get("micro_batch_size", reconstruction_batch_size)
    )
    if reconstruction_enabled:
        reconstruction_data_path = (
            args.data_path
            or reconstruction_cfg.get("data_path")
            or os.environ.get("DATA_PATH")
        )
        if not reconstruction_data_path:
            raise RuntimeError(
                "reconstruction.enabled is true, but no data path was provided. "
                "Set --data-path, reconstruction.data_path, or DATA_PATH."
            )
        reconstruction_loader, reconstruction_sampler = prepare_dataloader(
            reconstruction_data_path,
            int(reconstruction_cfg.get("image_size", 256)),
            reconstruction_batch_size,
            int(reconstruction_cfg.get("num_workers", 4)),
            rank,
            world_size,
            transform_type=int(reconstruction_cfg.get("transform_type", 0)),
            rrc_scale_min=float(reconstruction_cfg.get("rrc_scale_min", 0.8)),
            rrc_scale_max=float(reconstruction_cfg.get("rrc_scale_max", 1.0)),
        )
        reconstruction_batcher = ReconstructionBatcher(
            reconstruction_loader,
            reconstruction_sampler,
            device,
        )
        if rank == 0:
            log.info(
                f"[Reconstruction] enabled: data={reconstruction_data_path}, "
                f"batch/GPU={reconstruction_batch_size}, "
                f"micro_batch={reconstruction_micro_batch_size}, "
                f"weight={float(reconstruction_cfg.get('weight', 1.0))}"
            )

    max_train_steps = int(training_cfg.get("max_train_steps", 10000))
    grad_accum_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    if grad_accum_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive.")
    log_interval = int(training_cfg.get("log_interval", 10))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 1000))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    ema_decay = float(training_cfg.get("ema_decay", 0.999))
    num_groups = int(awm_cfg.get("classes_per_rank", 4))
    samples_per_group = int(awm_cfg.get("samples_per_class", 4))
    local_rollout_batch = num_groups * samples_per_group
    rollout_micro_batch_size = int(awm_cfg.get("rollout_micro_batch_size", local_rollout_batch))
    advantage_eps = float(awm_cfg.get("advantage_eps", 1e-4))
    advantage_clip = awm_cfg.get("advantage_clip", 5.0)
    advantage_clip = None if advantage_clip is None else float(advantage_clip)
    advantage_max = awm_cfg.get("advantage_max", 1.0)
    advantage_max = None if advantage_max is None else float(advantage_max)
    if num_groups <= 0 or samples_per_group <= 0:
        raise ValueError("awm.classes_per_rank and awm.samples_per_class must be positive.")
    if local_rollout_batch <= 0:
        raise ValueError("Local rollout batch must be positive.")
    reward_scale = float(reward_cfg.get("scale", 1.0))

    cfg_interval = tuple(float(x) for x in sample_cfg.get("cfg_interval", (0.1, 1.0)))
    cfg_scale = float(sample_cfg.get("cfg_scale", 1.0))
    cfg_norm_order = str(sample_cfg.get("cfg_norm_order", "norm_first"))
    grid_interval = int(sample_cfg.get("grid_interval", 0))

    device_type = "cuda" if device.type == "cuda" else "cpu"
    raw_module = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
    global_step = 0
    last_eval_step = None

    grid_labels = grid_noise = None
    grid_nrow = 0
    if is_main_process() and grid_interval > 0:
        grid_labels, grid_noise, grid_nrow = build_fixed_grid(ema_policy, sample_cfg, device)
        label_path = os.path.join(sample_dir, "fixed_grid_classes.txt")
        with open(label_path, "w") as f:
            f.write(" ".join(str(int(x)) for x in grid_labels.cpu().tolist()))
            f.write("\n")
        grid_path = save_sample_grid(
            ema_policy,
            sample_dir=sample_dir,
            step=global_step,
            labels=grid_labels,
            noise=grid_noise,
            nrow=grid_nrow,
            device=device,
            device_type=device_type,
            autocast_kwargs=autocast_kwargs,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            cfg_norm_order=cfg_norm_order,
        )
        logger.info(f"[Sample Grid] Saved: {grid_path}")

    if rank == 0:
        trainable = sum(p.numel() for p in params)
        log.info(
            f"Starting UNITE-AWM: steps={max_train_steps}, world_size={world_size}, "
            f"local_rollout_batch={local_rollout_batch}, "
            f"rollout_micro_batch={rollout_micro_batch_size}, "
            f"denoise_accum_steps={grad_accum_steps}, advantage_group_size={samples_per_group}, "
            f"trainable={trainable / 1e6:.2f}M"
        )

    while global_step < max_train_steps:
        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        metric_sums: Dict[str, torch.Tensor] = {}
        labels, _group_labels = sample_grouped_labels(
            num_groups=num_groups,
            samples_per_group=samples_per_group,
            num_classes=raw_module.policy.num_classes,
            device=device,
        )

        raw_module.policy.eval()
        z_clean, rewards = rollout_policy_batch(
            raw_module.policy,
            reward_model,
            labels=labels,
            reward_scale=reward_scale,
            rollout_micro_batch_size=rollout_micro_batch_size,
            device=device,
            device_type=device_type,
            autocast_kwargs=autocast_kwargs,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            cfg_norm_order=cfg_norm_order,
        )
        advantages = group_advantages(
            rewards,
            num_groups=num_groups,
            samples_per_group=samples_per_group,
            eps=advantage_eps,
            clip=advantage_clip,
            max_abs=advantage_max,
        )

        raw_module.policy.train()
        awm_chunks = split_loss_indices(labels.shape[0], grad_accum_steps, device)
        for accum_idx, awm_idx in enumerate(awm_chunks):
            awm_weight = awm_idx.numel() / labels.shape[0]
            sync_context = (
                ddp_model.no_sync()
                if hasattr(ddp_model, "no_sync") and accum_idx < len(awm_chunks) - 1
                else torch.enable_grad()
            )
            with sync_context:
                with autocast(device_type, **autocast_kwargs):
                    loss, metrics = ddp_model(
                        z_clean=z_clean[awm_idx],
                        labels=labels[awm_idx],
                        advantages=advantages[awm_idx],
                        awm_loss_weight=float(awm_weight),
                    )
                loss.backward()

            for key, value in metrics.items():
                metric_weight = metric_accum_weight(key, float(awm_weight), 0.0)
                metric_sums[key] = metric_sums.get(key, torch.zeros_like(value)) + value * metric_weight

        reconstruction_images = None
        reconstruction_chunks: list[torch.Tensor] = []
        if reconstruction_batcher is not None:
            reconstruction_images = reconstruction_batcher.next_images()
            reconstruction_chunks = split_indices_by_micro_batch_size(
                reconstruction_images.shape[0],
                reconstruction_micro_batch_size,
                device,
            )
        for accum_idx, recon_idx in enumerate(reconstruction_chunks):
            assert reconstruction_images is not None
            recon_weight = recon_idx.numel() / reconstruction_images.shape[0]
            sync_context = (
                ddp_model.no_sync()
                if hasattr(ddp_model, "no_sync") and accum_idx < len(reconstruction_chunks) - 1
                else torch.enable_grad()
            )
            with sync_context:
                with autocast(device_type, **autocast_kwargs):
                    loss, metrics = ddp_model(
                        recon_images=reconstruction_images[recon_idx],
                        recon_loss_weight=float(recon_weight),
                    )
                loss.backward()

            for key, value in metrics.items():
                metric_weight = metric_accum_weight(key, 0.0, float(recon_weight))
                metric_sums[key] = metric_sums.get(key, torch.zeros_like(value)) + value * metric_weight

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(params, clip_grad)
        optimizer.step()
        update_ema_policy(ema_policy, raw_module.policy, ema_decay)
        if raw_module.ema_beta > 0:
            update_ema_policy(
                kl_ema_policy,
                raw_module.policy,
                get_kl_ema_decay(awm_cfg, global_step),
            )

        completed_step = global_step + 1
        if (
            is_main_process()
            and grid_interval > 0
            and completed_step % grid_interval == 0
        ):
            grid_path = save_sample_grid(
                ema_policy,
                sample_dir=sample_dir,
                step=completed_step,
                labels=grid_labels,
                noise=grid_noise,
                nrow=grid_nrow,
                device=device,
                device_type=device_type,
                autocast_kwargs=autocast_kwargs,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_norm_order=cfg_norm_order,
            )
            logger.info(f"[Sample Grid] Saved: {grid_path}")

        if is_main_process() and global_step % log_interval == 0:
            log_stats = {key: value.item() for key, value in metric_sums.items()}
            log_stats["reward/mean"] = rewards.detach().mean().item()
            log_stats["awm/local_rollout_batch"] = float(local_rollout_batch)
            log_stats["awm/loss_microbatches"] = float(len(awm_chunks))
            if reconstruction_images is not None:
                log_stats["recon/batch"] = float(reconstruction_images.shape[0])
                log_stats["recon/microbatches"] = float(len(reconstruction_chunks))
            log_stats["perf/step_sec"] = time.time() - step_start
            logger.info(
                f"[Step {global_step}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in log_stats.items())
            )

        is_checkpoint_step = (
            checkpoint_interval > 0
            and completed_step > 0
            and completed_step % checkpoint_interval == 0
        )
        if is_checkpoint_step:
            if is_main_process():
                ckpt_path = f"{checkpoint_dir}/{completed_step:07d}.pt"
                save_awm_checkpoint(
                    ckpt_path,
                    step=completed_step,
                    awm_model=ddp_model,
                    ema_policy=ema_policy,
                    kl_ema_policy=kl_ema_policy,
                    optimizer=optimizer,
                    cfg_scale=cfg_scale,
                )
                logger.info(f"[Checkpoint] Saved: {ckpt_path}")
            if dist.is_initialized():
                dist.barrier()
            run_checkpoint_eval(
                ema_policy=ema_policy,
                reward_model=reward_model,
                eval_cfg=eval_cfg,
                sample_cfg=sample_cfg,
                device=device,
                global_step=completed_step,
                eval_dir=eval_dir,
                log=logger,
                autocast_kwargs=autocast_kwargs,
            )
            last_eval_step = completed_step
            raw_module.policy.train()

        global_step += 1

    if is_main_process():
        final_path = f"{checkpoint_dir}/final.pt"
        save_awm_checkpoint(
            final_path,
            step=global_step,
            awm_model=ddp_model,
            ema_policy=ema_policy,
            kl_ema_policy=kl_ema_policy,
            optimizer=optimizer,
            cfg_scale=cfg_scale,
        )
        logger.info(f"[Checkpoint] Saved final: {final_path}")
    if dist.is_initialized():
        dist.barrier()
    if last_eval_step != global_step:
        run_checkpoint_eval(
            ema_policy=ema_policy,
            reward_model=reward_model,
            eval_cfg=eval_cfg,
            sample_cfg=sample_cfg,
            device=device,
            global_step=global_step,
            eval_dir=eval_dir,
            log=logger,
            autocast_kwargs=autocast_kwargs,
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()
