import os
import sys
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# 确保从项目根目录运行脚本时，可以导入 src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.celeba_dataset import CelebAAttrDataset
from src.data.transforms import build_celeba_transform
from src.models.attr_classifier import build_attr_classifier


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_multilabel_accuracy(logits: torch.Tensor, targets: torch.Tensor):
    """
    多标签属性分类准确率。

    logits:  [B, K]
    targets: [B, K], 0/1

    返回:
        overall_acc: 所有属性位的平均准确率
        per_attr_acc: 每个属性各自的准确率, shape [K]
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    correct = (preds == targets).float()

    overall_acc = correct.mean().item()
    per_attr_acc = correct.mean(dim=0).detach().cpu()

    return overall_acc, per_attr_acc


def build_dataloader(cfg, split: str, batch_size: int, num_workers: int, mode: str):
    data_cfg = cfg["data"]

    transform = build_celeba_transform(
        image_size=data_cfg["image_size"],
        crop_size=data_cfg["crop_size"],
        mode=mode,
    )

    dataset = CelebAAttrDataset(
        root=data_cfg["celeba_root"],
        split=split,
        selected_attrs=data_cfg["selected_attrs"],
        transform=transform,
        image_dir=data_cfg["image_dir"],
        attr_file=data_cfg["attr_file"],
        partition_file=data_cfg["partition_file"],
    )

    shuffle = split == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataset, loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch: int,
    attr_names,
    log_interval: int = 10,
    max_batches: int | None = None,
):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)

    for batch_idx, (images, attrs, filenames) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        attrs = attrs.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, attrs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc, per_attr_acc = compute_multilabel_accuracy(logits, attrs)

        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}",
            })

    avg_loss = total_loss / max(total_batches, 1)
    avg_acc = total_acc / max(total_batches, 1)

    return avg_loss, avg_acc


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    attr_names,
    split_name: str = "valid",
    max_batches: int | None = None,
):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    per_attr_acc_sum = torch.zeros(len(attr_names))

    pbar = tqdm(loader, desc=f"Eval {split_name}", ncols=100)

    for batch_idx, (images, attrs, filenames) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        attrs = attrs.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, attrs)

        acc, per_attr_acc = compute_multilabel_accuracy(logits, attrs)

        total_loss += loss.item()
        total_acc += acc
        per_attr_acc_sum += per_attr_acc
        total_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
        })

    avg_loss = total_loss / max(total_batches, 1)
    avg_acc = total_acc / max(total_batches, 1)
    avg_per_attr_acc = per_attr_acc_sum / max(total_batches, 1)

    return avg_loss, avg_acc, avg_per_attr_acc


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    cfg,
    attr_names,
    val_loss,
    val_acc,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "attr_names": attr_names,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/mvp_64.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoints/attr_classifier")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_interval", type=int, default=10)

    # 本地 smoke test 用：只跑几个 batch，确认流程没问题
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("train", {}).get("seed", 42)
    set_seed(seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = cfg.get("train", {}).get("batch_size", 8)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = cfg.get("train", {}).get("num_workers", 0)

    attr_names = cfg["data"]["selected_attrs"]
    num_attrs = len(attr_names)

    print("=" * 80)
    print("Phase 1: Train Attribute Classifier")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Attributes: {attr_names}")
    print(f"Num attrs: {num_attrs}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Pretrained: {args.pretrained}")
    print("=" * 80)

    train_dataset, train_loader = build_dataloader(
        cfg=cfg,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        mode="train",
    )

    valid_dataset, valid_loader = build_dataloader(
        cfg=cfg,
        split="valid",
        batch_size=batch_size,
        num_workers=num_workers,
        mode="eval",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    model = build_attr_classifier(
        num_attrs=num_attrs,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            attr_names=attr_names,
            log_interval=args.log_interval,
            max_batches=args.max_train_batches,
        )

        val_loss, val_acc, val_per_attr_acc = validate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            attr_names=attr_names,
            split_name="valid",
            max_batches=args.max_val_batches,
        )

        elapsed = time.time() - start_time

        print("-" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Valid loss: {val_loss:.4f} | Valid acc: {val_acc:.4f}")
        print("Per-attribute valid accuracy:")
        for name, acc in zip(attr_names, val_per_attr_acc.tolist()):
            print(f"  {name}: {acc:.4f}")
        print(f"Time: {elapsed:.2f}s")
        print("-" * 80)

        last_path = os.path.join(args.output_dir, "last.pt")
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            attr_names=attr_names,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.output_dir, "best.pt")
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                attr_names=attr_names,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            print(f"Saved best checkpoint to: {best_path}")

    print("=" * 80)
    print("Training finished.")
    print(f"Best valid acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()