import os
import sys
import json
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.text.templates import (
    generate_instruction_samples,
    build_vocab,
    split_samples,
    TextInstructionDataset,
    ID_TO_ACTION,
)
from src.models.text_encoder import build_text_encoder


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_action_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """
    logits: [B, K, 3]
    labels: [B, K]

    返回:
        overall_acc: 所有属性动作的平均准确率
        per_attr_acc: 每个属性动作预测准确率, [K]
    """
    preds = torch.argmax(logits, dim=-1)  # [B, K]
    correct = (preds == labels).float()

    overall_acc = correct.mean().item()
    per_attr_acc = correct.mean(dim=0).detach().cpu()

    return overall_acc, per_attr_acc


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, criterion):
    """
    多属性三分类损失。

    logits: [B, K, 3]
    labels: [B, K]

    将其展平成:
        [B*K, 3] vs [B*K]
    """
    bsz, num_attrs, num_classes = logits.shape
    loss = criterion(
        logits.reshape(bsz * num_attrs, num_classes),
        labels.reshape(bsz * num_attrs),
    )
    return loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch: int,
):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, labels, criterion)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc, _ = compute_action_accuracy(logits, labels)

        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
        })

    return (
        total_loss / max(total_batches, 1),
        total_acc / max(total_batches, 1),
    )


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    attr_names,
):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    per_attr_acc_sum = torch.zeros(len(attr_names))

    all_examples = []

    pbar = tqdm(loader, desc="Eval valid", ncols=100)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        texts = batch["text"]

        logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, labels, criterion)

        acc, per_attr_acc = compute_action_accuracy(logits, labels)

        total_loss += loss.item()
        total_acc += acc
        per_attr_acc_sum += per_attr_acc
        total_batches += 1

        preds = torch.argmax(logits, dim=-1).detach().cpu()

        for i in range(min(3, len(texts))):
            all_examples.append({
                "text": texts[i],
                "label": labels[i].detach().cpu().tolist(),
                "pred": preds[i].tolist(),
            })

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
        })

    avg_loss = total_loss / max(total_batches, 1)
    avg_acc = total_acc / max(total_batches, 1)
    avg_per_attr_acc = per_attr_acc_sum / max(total_batches, 1)

    return avg_loss, avg_acc, avg_per_attr_acc, all_examples[:5]


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    cfg,
    vocab,
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
        "vocab": vocab,
        "attr_names": attr_names,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/mvp_64.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoints/text_encoder")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--device", type=str, default="auto")

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

    attr_names = cfg["data"]["selected_attrs"]
    num_attrs = len(attr_names)

    text_cfg = cfg["text"]
    max_len = text_cfg["max_len"]

    # 1. 自动生成模板数据
    all_samples = generate_instruction_samples(attr_names)

    train_samples, valid_samples = split_samples(
        all_samples,
        train_ratio=text_cfg.get("train_ratio", 0.8),
        seed=seed,
    )

    # 2. 构造词表
    vocab = build_vocab(train_samples + valid_samples)

    # 3. 构造 Dataset / DataLoader
    train_dataset = TextInstructionDataset(
        samples=train_samples,
        vocab=vocab,
        max_len=max_len,
    )

    valid_dataset = TextInstructionDataset(
        samples=valid_samples,
        vocab=vocab,
        max_len=max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4. 构造模型
    model = build_text_encoder(
        vocab_size=len(vocab),
        num_attrs=num_attrs,
        max_len=max_len,
        embed_dim=text_cfg["embed_dim"],
        num_heads=text_cfg["num_heads"],
        num_layers=text_cfg["num_layers"],
        ff_dim=text_cfg["ff_dim"],
        dropout=text_cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 保存词表，后面 infer.py 会用
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Phase 2: Train Text Instruction Encoder")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Attributes: {attr_names}")
    print(f"Num attrs: {num_attrs}")
    print(f"Num total samples: {len(all_samples)}")
    print(f"Num train samples: {len(train_samples)}")
    print(f"Num valid samples: {len(valid_samples)}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Max len: {max_len}")
    print(f"Vocab saved to: {vocab_path}")
    print("=" * 80)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_loss, val_acc, val_per_attr_acc, examples = validate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            attr_names=attr_names,
        )

        elapsed = time.time() - start_time

        print("-" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Valid loss: {val_loss:.4f} | Valid acc: {val_acc:.4f}")

        print("Per-attribute valid action accuracy:")
        for name, acc in zip(attr_names, val_per_attr_acc.tolist()):
            print(f"  {name}: {acc:.4f}")

        print("Example predictions:")
        for ex in examples:
            label_names = [ID_TO_ACTION[x] for x in ex["label"]]
            pred_names = [ID_TO_ACTION[x] for x in ex["pred"]]
            print(f"  text:  {ex['text']}")
            print(f"  label: {label_names}")
            print(f"  pred:  {pred_names}")

        print(f"Time: {elapsed:.2f}s")
        print("-" * 80)

        last_path = os.path.join(args.output_dir, "last.pt")
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            vocab=vocab,
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
                vocab=vocab,
                attr_names=attr_names,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            print(f"Saved best checkpoint to: {best_path}")

    print("=" * 80)
    print("Text encoder training finished.")
    print(f"Best valid acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()