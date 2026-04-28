import os
import time
import random
from dataclasses import dataclass, asdict
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from nnue import NNUE, NNUELoss, clamp_weights
from dataset_io import load_training_shard
from data_handling import make_sparse_batch_from_preprocessed
from nnue_constants import NUM_FEATURES, BUCKET_NB


Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class TrainConfig:
    model_dir: str = "models"
    model_name: str = "nnue_model_bucket_v8"
    load_existing: bool = True
    keep_optimizer_state: bool = True

    lr: float = 0.0005
    lr_gamma: float = 0.98
    batch_size: int = 32768
    scaling_factor: float = 410.0
    exponent: float = 2.5

    validate_every_shards: int = 10
    train_shard_count: int = 564
    validation_shard_count: int = 2


def get_model_paths(cfg: TrainConfig) -> dict:
    os.makedirs(cfg.model_dir, exist_ok=True)

    base = os.path.join(cfg.model_dir, cfg.model_name)

    return {
        "latest": base + "_latest.pt",
        "best": base + "_best.pt",
        "final": base + "_final.pt",
    }


def save_training_state(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    epoch: int,
    completed_shards: int,
    best_val_loss: float,
    cfg: TrainConfig,
):
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "completed_shards": completed_shards,
        "best_val_loss": best_val_loss,
        "config": asdict(cfg),
    }

    if cfg.keep_optimizer_state:
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, path)


def load_model_or_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.StepLR | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        return state

    model.load_state_dict(state)
    return {
        "epoch": 0,
        "completed_shards": 0,
        "best_val_loss": float("inf"),
    }


def init_weights(layer: torch.nn.Module):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def maybe_pin_batch(batch: Batch) -> Batch:
    return tuple(
        x.pin_memory() if torch.is_tensor(x) and x.device.type == "cpu" else x
        for x in batch
    )  # type: ignore[return-value]


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return tuple(
        x.to(device, non_blocking=True) if torch.is_tensor(x) else x
        for x in batch
    )  # type: ignore[return-value]


def build_shard_batches_cpu(
    shard: dict,
    batch_size: int,
    max_features: int,
    pin_memory: bool,
    shuffle: bool,
) -> List[Batch]:
    num_positions = len(shard["stm"])
    order = np.arange(num_positions)

    if shuffle:
        np.random.shuffle(order)

    batches: List[Batch] = []

    for start in range(0, num_positions, batch_size):
        rows = order[start:start + batch_size]

        batch = make_sparse_batch_from_preprocessed(
            shard,
            rows,
            max_features=max_features,
            device=torch.device("cpu"),
        )

        if pin_memory:
            batch = maybe_pin_batch(batch)

        batches.append(batch)

    return batches


def load_validation_batches(
    val_shard_paths: Sequence[str],
    batch_size: int,
    max_features: int,
    pin_memory: bool,
) -> List[Batch]:
    print("loading validation shard(s)...")

    val_batches: List[Batch] = []

    for path in val_shard_paths:
        shard = load_training_shard(path)

        val_batches.extend(
            build_shard_batches_cpu(
                shard=shard,
                batch_size=batch_size,
                max_features=max_features,
                pin_memory=pin_memory,
                shuffle=False,
            )
        )

    print(f"cached {len(val_batches)} validation batches in RAM")
    return val_batches


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    val_batches: Sequence[Batch],
    loss_fn: NNUELoss,
    device: torch.device,
    bucket_nb: int,
) -> dict:
    model.eval()

    total_loss_sum = 0.0
    total_samples = 0

    bucket_loss_sum = np.zeros(bucket_nb, dtype=np.float64)
    bucket_count = np.zeros(bucket_nb, dtype=np.int64)

    start = time.perf_counter()

    for batch_cpu in val_batches:
        white_features_t, black_features_t, stm_t, score_t, bucket_t = move_batch_to_device(
            batch_cpu,
            device,
        )

        output = model(white_features_t, black_features_t, stm_t, bucket_t)
        per_sample = loss_fn.per_sample(output, score_t).view(-1)
        buckets = bucket_t.view(-1)

        total_loss_sum += per_sample.sum().item()
        total_samples += per_sample.numel()

        bucket_count_batch = torch.bincount(buckets, minlength=bucket_nb)

        bucket_loss_batch = torch.zeros(bucket_nb, device=device)
        bucket_loss_batch.scatter_add_(0, buckets, per_sample)

        bucket_count += bucket_count_batch.cpu().numpy()
        bucket_loss_sum += bucket_loss_batch.cpu().numpy()

    cuda_sync(device)

    elapsed = time.perf_counter() - start
    val_loss = total_loss_sum / max(total_samples, 1)
    bucket_loss = bucket_loss_sum / np.maximum(bucket_count, 1)

    return {
        "val_loss": val_loss,
        "bucket_loss": bucket_loss,
        "bucket_count": bucket_count,
        "elapsed": elapsed,
        "positions_per_sec": total_samples / max(elapsed, 1e-9),
    }


def train_batches(
    model: torch.nn.Module,
    optimizer: AdamW,
    loss_fn: NNUELoss,
    batches: Sequence[Batch],
    device: torch.device,
) -> dict:
    model.train()

    total_loss = 0.0
    batch_count = 0
    positions = 0

    start = time.perf_counter()

    for batch_cpu in batches:
        white_features_t, black_features_t, stm_t, score_t, bucket_t = move_batch_to_device(
            batch_cpu,
            device,
        )

        output = model(white_features_t, black_features_t, stm_t, bucket_t)
        loss = loss_fn(output, score_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        clamp_weights(model)

        batch_pos = score_t.numel()
        total_loss += loss.item()
        batch_count += 1
        positions += batch_pos

    cuda_sync(device)

    elapsed = time.perf_counter() - start

    return {
        "loss_sum": total_loss,
        "loss_avg": total_loss / max(batch_count, 1),
        "batch_count": batch_count,
        "positions": positions,
        "elapsed": elapsed,
        "pos_per_sec": positions / max(elapsed, 1e-9),
        "batches_per_sec": batch_count / max(elapsed, 1e-9),
    }


def print_bucket_metrics(val_metrics: dict, bucket_nb: int):
    parts = []

    for b in range(bucket_nb):
        parts.append(
            f"b{b}:{val_metrics['bucket_loss'][b]:.6f}"
            f"({val_metrics['bucket_count'][b]})"
        )

    print("[validation buckets] " + " ".join(parts))


def handle_validation_and_checkpoint(
    cfg: TrainConfig,
    model_paths: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    epoch: int,
    completed_shards: int,
    best_val_loss: float,
    val_metrics: dict,
    train_loss_avg: float,
    train_pos_per_sec: float,
    train_batches_per_sec: float,
) -> float:
    gap = val_metrics["val_loss"] - train_loss_avg

    print(
        f"[validation] epoch={epoch} shard={completed_shards} "
        f"train_loss={train_loss_avg:.6f} "
        f"val_loss={val_metrics['val_loss']:.6f} "
        f"gap={gap:.6f} "
        f"train_pos_per_sec={train_pos_per_sec:.0f} "
        f"train_batches_per_sec={train_batches_per_sec:.2f} "
        f"val_pos_per_sec={val_metrics['positions_per_sec']:.0f}"
    )

    print_bucket_metrics(val_metrics, BUCKET_NB)

    save_training_state(
        path=model_paths["latest"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        completed_shards=completed_shards,
        best_val_loss=best_val_loss,
        cfg=cfg,
    )

    print(f"[save] latest saved to {model_paths['latest']}")

    if val_metrics["val_loss"] < best_val_loss:
        best_val_loss = val_metrics["val_loss"]

        save_training_state(
            path=model_paths["best"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            completed_shards=completed_shards,
            best_val_loss=best_val_loss,
            cfg=cfg,
        )

        print(
            f"[save] new best saved to {model_paths['best']} "
            f"val_loss={best_val_loss:.6f}"
        )

    return best_val_loss


def main():
    cfg = TrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model_paths = get_model_paths(cfg)

    model = NNUE(bucket_nb=BUCKET_NB).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=0,
    )

    scheduler = StepLR(
        optimizer,
        step_size=1,
        gamma=cfg.lr_gamma,
    )

    best_val_loss = float("inf")

    if cfg.load_existing and os.path.exists(model_paths["best"]):
        state = load_model_or_checkpoint(
            path=model_paths["best"],
            model=model,
            optimizer=optimizer if cfg.keep_optimizer_state else None,
            scheduler=scheduler if cfg.keep_optimizer_state else None,
            device=device,
        )

        best_val_loss = state.get("best_val_loss", float("inf"))
        clamp_weights(model)

        print(f"loaded existing checkpoint from {model_paths['best']}")
        print(f"best_val_loss={best_val_loss:.6f}")

    else:
        model.apply(init_weights)
        clamp_weights(model)
        print("initialized new model")

    loss_fn = NNUELoss(
        scaling_factor=cfg.scaling_factor,
        exponent=cfg.exponent,
    )

    all_shard_paths = [
        f"train_shards/shard{i}.npz"
        for i in range(cfg.train_shard_count)
    ]

    val_shard_paths = all_shard_paths[-cfg.validation_shard_count:]
    train_shard_paths = all_shard_paths[:-cfg.validation_shard_count]

    val_batches = load_validation_batches(
        val_shard_paths=val_shard_paths,
        batch_size=cfg.batch_size,
        max_features=NUM_FEATURES,
        pin_memory=pin_memory,
    )

    epoch = 0
    completed_shards = 0

    try:
        while True:
            epoch += 1
            random.shuffle(train_shard_paths)

            print(f"starting epoch {epoch}")

            running_train_loss_sum = 0.0
            running_train_batches = 0
            running_train_positions = 0
            segment_start_time = time.perf_counter()

            for completed_shards, shard_path in enumerate(train_shard_paths, start=1):
                shard = load_training_shard(shard_path)

                batches = build_shard_batches_cpu(
                    shard=shard,
                    batch_size=cfg.batch_size,
                    max_features=NUM_FEATURES,
                    pin_memory=pin_memory,
                    shuffle=True,
                )

                train_metrics = train_batches(
                    model=model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    batches=batches,
                    device=device,
                )

                running_train_loss_sum += train_metrics["loss_sum"]
                running_train_batches += train_metrics["batch_count"]
                running_train_positions += train_metrics["positions"]

                print(
                    f"{os.path.basename(shard_path)} "
                    f"train_loss={train_metrics['loss_avg']:.6f} "
                    f"train_pos_per_sec={train_metrics['pos_per_sec']:.0f} "
                    f"train_batches_per_sec={train_metrics['batches_per_sec']:.2f}"
                )

                if completed_shards % cfg.validate_every_shards == 0:
                    train_elapsed = time.perf_counter() - segment_start_time
                    train_loss_avg = running_train_loss_sum / max(running_train_batches, 1)
                    train_pos_per_sec = running_train_positions / max(train_elapsed, 1e-9)
                    train_batches_per_sec = running_train_batches / max(train_elapsed, 1e-9)

                    val_metrics = evaluate_validation(
                        model=model,
                        val_batches=val_batches,
                        loss_fn=loss_fn,
                        device=device,
                        bucket_nb=BUCKET_NB,
                    )

                    best_val_loss = handle_validation_and_checkpoint(
                        cfg=cfg,
                        model_paths=model_paths,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        completed_shards=completed_shards,
                        best_val_loss=best_val_loss,
                        val_metrics=val_metrics,
                        train_loss_avg=train_loss_avg,
                        train_pos_per_sec=train_pos_per_sec,
                        train_batches_per_sec=train_batches_per_sec,
                    )

                    running_train_loss_sum = 0.0
                    running_train_batches = 0
                    running_train_positions = 0
                    segment_start_time = time.perf_counter()

            print("saving model")

            save_training_state(
                path=model_paths["latest"],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                completed_shards=len(train_shard_paths),
                best_val_loss=best_val_loss,
                cfg=cfg,
            )

            print("saving completed")

            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"learning_rate={current_lr:.8g}")

    except KeyboardInterrupt:
        print("\nInterrupted by user, saving model...")

    finally:
        save_training_state(
            path=model_paths["final"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            completed_shards=completed_shards,
            best_val_loss=best_val_loss,
            cfg=cfg,
        )

        print(f"final save completed: {model_paths['final']}")


if __name__ == "__main__":
    main()
