#!/usr/bin/env python
"""Offline distributed FID/IS evaluation for saved UNITE-AWM checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml

from engines.eval import evaluate_fid
from models.unite import UNITE
from utils.distributed import cleanup_distributed, is_main_process, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved UNITE-AWM checkpoints.")
    parser.add_argument("--config", type=str, required=True, help="YAML config used to build UNITE.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory containing checkpoints.")
    parser.add_argument("--ckpt-glob", type=str, default="*.pt", help="Glob used inside --checkpoint-dir.")
    parser.add_argument("--ckpts", nargs="*", default=None, help="Explicit checkpoint paths.")
    parser.add_argument("--state-key", type=str, default="ema", help="Checkpoint state key to evaluate.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for metrics.jsonl/log.txt.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip checkpoints already in metrics.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Only list checkpoints that would be evaluated.")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cfg-scales", type=float, nargs="*", default=None)
    parser.add_argument("--cfg-intervals", type=str, nargs="*", default=None, help="Intervals like 0.1,1.0")
    parser.add_argument("--cfg-norm-orders", type=str, nargs="*", default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("eval_awm_checkpoints")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(output_dir / "log.txt")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def normalize_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("module.")
        new_key = new_key.replace("._orig_mod.", ".")
        new_key = new_key.removeprefix("_orig_mod.")
        normalized[new_key] = value
    return normalized


def contains_tensor_state_dict(value: Any) -> bool:
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

    valid = contains_tensor_state_dict(weights)
    if not valid and isinstance(weights, dict):
        valid = any(contains_tensor_state_dict(item) for item in weights.values())
    if not valid:
        kind = type(weights).__name__
        keys = list(weights.keys())[:10] if isinstance(weights, dict) else None
        raise RuntimeError(
            "INCEPTION_WEIGHTS must be a torch checkpoint containing a tensor state_dict; "
            f"got type={kind}, keys={keys} from {path}"
        )


def validate_eval_files() -> None:
    required = ("INCEPTION_WEIGHTS", "IN256_FID_STATS")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    for name in required:
        path = os.environ[name]
        if not os.path.isfile(path):
            raise RuntimeError(f"{name} does not point to a file: {path}")

    validate_inception_weights(os.environ["INCEPTION_WEIGHTS"])

    stats_path = os.environ["IN256_FID_STATS"]
    try:
        with np.load(stats_path) as stats:
            keys = set(stats.files)
    except Exception as exc:
        raise RuntimeError(f"Failed to read IN256_FID_STATS as an .npz file: {stats_path}") from exc
    if not {"mu", "sigma"}.issubset(keys):
        raise RuntimeError(f"IN256_FID_STATS must contain 'mu' and 'sigma', got keys={sorted(keys)}")


def checkpoint_sort_key(path: Path) -> Tuple[int, int | str, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.name)
    if stem == "final":
        return (2, 0, path.name)
    return (1, stem, path.name)


def resolve_checkpoints(args: argparse.Namespace) -> List[Path]:
    if args.ckpts:
        paths = [Path(x).expanduser() for x in args.ckpts]
    elif args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).expanduser()
        paths = sorted(checkpoint_dir.glob(args.ckpt_glob), key=checkpoint_sort_key)
    else:
        raise ValueError("Provide either --checkpoint-dir or --ckpts.")

    paths = [p.resolve() for p in paths if p.is_file()]
    if not paths:
        raise ValueError("No checkpoint files found.")
    return paths


def parse_cfg_intervals(raw: Iterable[str] | None, fallback: Any) -> List[Tuple[float, float]]:
    if raw is not None:
        intervals = []
        for item in raw:
            parts = [part.strip() for part in item.split(",")]
            if len(parts) != 2:
                raise ValueError(f"Invalid --cfg-intervals item: {item}")
            intervals.append((float(parts[0]), float(parts[1])))
        return intervals

    if fallback is None:
        return [(0.1, 1.0)]
    if (
        isinstance(fallback, (list, tuple))
        and len(fallback) == 2
        and all(isinstance(x, (int, float)) for x in fallback)
    ):
        fallback = [fallback]
    return [(float(item[0]), float(item[1])) for item in fallback]


def list_from_config(cli_value: Any, config_value: Any, fallback: Any) -> List[Any]:
    value = cli_value if cli_value is not None and len(cli_value) > 0 else config_value
    if value is None:
        value = fallback
    if isinstance(value, (str, int, float)):
        value = [value]
    return list(value)


def load_checkpoint_state(model: UNITE, path: Path, state_key: str) -> int:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Expected dict checkpoint, got {type(checkpoint).__name__}: {path}")
    if state_key not in checkpoint:
        raise RuntimeError(
            f"Checkpoint {path} does not contain state key '{state_key}'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    state = checkpoint[state_key]
    msg = model.load_state_dict(normalize_checkpoint_keys(state), strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint does not match UNITE architecture: "
            f"missing={msg.missing_keys}, unexpected={msg.unexpected_keys}"
        )

    step = checkpoint.get("step")
    if step is None and path.stem.isdigit():
        step = int(path.stem)
    if step is None:
        step = -1
    del checkpoint, state
    return int(step)


def existing_checkpoint_names(metrics_path: Path, state_key: str) -> set[str]:
    if not metrics_path.exists():
        return set()
    seen = set()
    with open(metrics_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("state_key") == state_key and "checkpoint_name" in item:
                seen.add(str(item["checkpoint_name"]))
    return seen


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()

    with open(args.config, "r") as f:
        full_cfg = yaml.safe_load(f)
    training_cfg = dict(full_cfg.get("training", {}))
    gen_tok_cfg = dict(full_cfg.get("gen_tok", {}))
    eval_cfg = dict(full_cfg.get("eval", {}))
    sample_cfg = dict(full_cfg.get("sampling", {}))

    paths = resolve_checkpoints(args)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif args.checkpoint_dir:
        output_dir = (Path(args.checkpoint_dir).expanduser().resolve().parent / "eval_offline")
    else:
        output_dir = (paths[0].parent.parent / "eval_offline").resolve()

    logger = setup_logger(output_dir)
    if is_main_process():
        logger.info(f"Found {len(paths)} checkpoints.")
        for path in paths:
            logger.info(f"[Eval Queue] {path}")
    if args.dry_run:
        cleanup_distributed()
        return

    validate_eval_files()

    precision = args.precision or str(training_cfg.get("precision", "bf16"))
    if precision == "bf16":
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    elif precision == "fp16":
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    else:
        autocast_kwargs = dict(enabled=False)

    cfg_scales = [float(x) for x in list_from_config(
        args.cfg_scales,
        eval_cfg.get("cfg_scales"),
        [float(sample_cfg.get("cfg_scale", 1.0))],
    )]
    cfg_intervals = parse_cfg_intervals(
        args.cfg_intervals,
        eval_cfg.get("cfg_intervals", [sample_cfg.get("cfg_interval", (0.1, 1.0))]),
    )
    cfg_norm_orders = [str(x) for x in list_from_config(
        args.cfg_norm_orders,
        eval_cfg.get("cfg_norm_orders"),
        [str(sample_cfg.get("cfg_norm_order", "norm_first"))],
    )]
    num_images = int(args.num_images or eval_cfg.get("num_images", 50000))
    num_classes = int(args.num_classes or eval_cfg.get("num_classes", training_cfg.get("num_classes", 1000)))
    image_size = int(args.image_size or eval_cfg.get("image_size", 256))
    batch_size = int(args.batch_size or eval_cfg.get("batch_size", 50))

    model = UNITE(gen_tok_cfg, num_classes=int(training_cfg.get("num_classes", 1000))).to(device)
    model.eval().requires_grad_(False)

    metrics_path = output_dir / "metrics.jsonl"
    start_time = time.time()
    for ckpt_path in paths:
        skip = False
        if is_main_process() and args.skip_existing:
            skip = ckpt_path.name in existing_checkpoint_names(metrics_path, args.state_key)
        if dist.is_initialized():
            skip_t = torch.tensor([int(skip)], device=device)
            dist.broadcast(skip_t, src=0)
            skip = bool(skip_t.item())
        if skip:
            if is_main_process():
                logger.info(f"[Skip] {ckpt_path.name} already exists in {metrics_path}")
            continue

        if is_main_process():
            logger.info(f"[Load] {ckpt_path}")
        step = load_checkpoint_state(model, ckpt_path, args.state_key)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

        eval_start = time.time()
        best_cfg, fid, is_score = evaluate_fid(
            model,
            device=device,
            global_step=step,
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

        if is_main_process():
            metrics = {
                "checkpoint_name": ckpt_path.name,
                "checkpoint_path": str(ckpt_path),
                "state_key": args.state_key,
                "step": int(step),
                "eval/fid": float(fid),
                "eval/is": float(is_score),
                "eval/best_cfg_scale": float(best_cfg),
                "eval/num_images": int(num_images),
                "eval/batch_size": int(batch_size),
                "eval/sec": time.time() - eval_start,
            }
            if num_images == 50000:
                metrics["eval/fid50k"] = float(fid)
                metrics["eval/is50k"] = float(is_score)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics, sort_keys=True) + "\n")
            logger.info(
                f"[Done] {ckpt_path.name}: FID={fid:.4f}, IS={is_score:.4f}, "
                f"best_cfg={best_cfg:.2f}, step={step}"
            )
        if dist.is_initialized():
            dist.barrier()

    if is_main_process():
        logger.info(f"Finished offline eval in {(time.time() - start_time) / 60:.2f} min. Metrics: {metrics_path}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
