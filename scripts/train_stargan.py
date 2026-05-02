import os
import sys
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.celeba_dataset import CelebAAttrDataset
from src.data.transforms import build_celeba_transform
from src.models.stargan import (
    build_stargan_generator,
    build_stargan_discriminator,
)


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(x: torch.Tensor):
    """
    [-1, 1] -> [0, 1]
    """
    return (x + 1.0) / 2.0


def sample_target_labels(c_org: torch.Tensor):
    """
    随机采样目标属性向量 c_trg。

    c_org: [B, K], 0/1

    返回:
        c_trg: [B, K], 0/1

    为了避免目标属性和原属性完全一样，如果某个样本采样后完全相同，
    就随机翻转其中一个属性。
    """
    c_trg = torch.randint(
        low=0,
        high=2,
        size=c_org.shape,
        device=c_org.device,
        dtype=c_org.dtype,
    )

    same = (c_trg == c_org).all(dim=1)

    if same.any():
        same_indices = same.nonzero(as_tuple=False).view(-1)
        num_attrs = c_org.size(1)

        for idx in same_indices:
            j = torch.randint(0, num_attrs, size=(1,), device=c_org.device).item()
            c_trg[idx, j] = 1.0 - c_trg[idx, j]

    return c_trg


def gradient_penalty(D, x_real, x_fake, device):
    """
    WGAN-GP 梯度惩罚：

        E[(||∇_x D(x_hat)||_2 - 1)^2]

    这里 D_src 是 PatchGAN 输出，所以先对所有 patch 分数求和/均值都可以。
    用 autograd 对输入图像求梯度。
    """
    batch_size = x_real.size(0)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    x_hat = alpha * x_real + (1.0 - alpha) * x_fake
    x_hat.requires_grad_(True)

    out_src, _ = D(x_hat)

    grad_outputs = torch.ones_like(out_src, device=device)

    gradients = torch.autograd.grad(
        outputs=out_src,
        inputs=x_hat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    return gp


def build_dataloader(cfg, split: str, batch_size: int, num_workers: int):
    data_cfg = cfg["data"]

    transform = build_celeba_transform(
        image_size=data_cfg["image_size"],
        crop_size=data_cfg["crop_size"],
        mode="train" if split == "train" else "eval",
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

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return dataset, loader


@torch.no_grad()
def save_debug_samples(G, fixed_images, fixed_attrs, attr_names, epoch, step, output_dir, device):
    """
    保存可视化样例。

    对固定图像生成多种目标属性：
        1. 原图
        2. Smiling = 1
        3. Smiling = 0
        4. Eyeglasses = 1
        5. Eyeglasses = 0
        6. 全属性翻转
    """
    os.makedirs(output_dir, exist_ok=True)

    G.eval()

    fixed_images = fixed_images.to(device)
    fixed_attrs = fixed_attrs.to(device)

    images_to_save = [fixed_images.cpu()]

    num_attrs = fixed_attrs.size(1)

    target_list = []

    # 单属性设置
    for attr_idx in range(num_attrs):
        c_set1 = fixed_attrs.clone()
        c_set1[:, attr_idx] = 1.0
        target_list.append(c_set1)

        c_set0 = fixed_attrs.clone()
        c_set0[:, attr_idx] = 0.0
        target_list.append(c_set0)

    # 全属性翻转
    c_flip = 1.0 - fixed_attrs
    target_list.append(c_flip)

    for c_trg in target_list:
        x_fake = G(fixed_images, c_trg)
        images_to_save.append(x_fake.cpu())

    grid = torch.cat(images_to_save, dim=0)
    grid = denormalize(grid)

    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}_step_{step:06d}.jpg")
    save_image(grid, save_path, nrow=fixed_images.size(0))

    G.train()

    return save_path


def save_checkpoint(path, G, D, g_optimizer, d_optimizer, epoch, step, cfg, attr_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "step": step,
        "G_state_dict": G.state_dict(),
        "D_state_dict": D.state_dict(),
        "g_optimizer_state_dict": g_optimizer.state_dict(),
        "d_optimizer_state_dict": d_optimizer.state_dict(),
        "config": cfg,
        "attr_names": attr_names,
    }

    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/mvp_64.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoints/stargan")
    parser.add_argument("--sample_dir", type=str, default="outputs/samples/stargan")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--device", type=str, default="auto")

    # 本地 smoke test 用
    parser.add_argument("--max_batches", type=int, default=None)

    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)

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

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    stargan_cfg = cfg["stargan"]

    attr_names = data_cfg["selected_attrs"]
    c_dim = len(attr_names)
    image_size = data_cfg["image_size"]

    batch_size = args.batch_size if args.batch_size is not None else train_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else train_cfg["num_workers"]

    train_dataset, train_loader = build_dataloader(
        cfg=cfg,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    G = build_stargan_generator(
        c_dim=c_dim,
        g_conv_dim=stargan_cfg["g_conv_dim"],
        g_repeat_num=stargan_cfg["g_repeat_num"],
    ).to(device)

    D = build_stargan_discriminator(
        image_size=image_size,
        c_dim=c_dim,
        d_conv_dim=stargan_cfg["d_conv_dim"],
        d_repeat_num=stargan_cfg["d_repeat_num"],
    ).to(device)

    g_optimizer = torch.optim.Adam(
        G.parameters(),
        lr=stargan_cfg["lr"],
        betas=(stargan_cfg["beta1"], stargan_cfg["beta2"]),
    )

    d_optimizer = torch.optim.Adam(
        D.parameters(),
        lr=stargan_cfg["lr"],
        betas=(stargan_cfg["beta1"], stargan_cfg["beta2"]),
    )

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    lambda_cls = stargan_cfg["lambda_cls"]
    lambda_rec = stargan_cfg["lambda_rec"]
    lambda_gp = stargan_cfg["lambda_gp"]
    n_critic = stargan_cfg["n_critic"]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    fixed_images, fixed_attrs, _ = next(iter(train_loader))
    fixed_images = fixed_images[: min(4, fixed_images.size(0))]
    fixed_attrs = fixed_attrs[: min(4, fixed_attrs.size(0))]

    print("=" * 80)
    print("Phase 3: Train StarGAN")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Image size: {image_size}")
    print(f"Attributes: {attr_names}")
    print(f"c_dim: {c_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"lambda_cls: {lambda_cls}")
    print(f"lambda_rec: {lambda_rec}")
    print(f"lambda_gp: {lambda_gp}")
    print(f"n_critic: {n_critic}")
    print("=" * 80)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)

        for batch_idx, (x_real, c_org, filenames) in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            global_step += 1

            x_real = x_real.to(device, non_blocking=True)
            c_org = c_org.to(device, non_blocking=True)

            c_trg = sample_target_labels(c_org)

            # ============================================================
            # 1. Train Discriminator
            # ============================================================
            with torch.no_grad():
                x_fake = G(x_real, c_trg)

            out_src_real, out_cls_real = D(x_real)
            out_src_fake, _ = D(x_fake.detach())

            d_loss_real = -torch.mean(out_src_real)
            d_loss_fake = torch.mean(out_src_fake)
            d_loss_cls = bce(out_cls_real, c_org)

            gp = gradient_penalty(D, x_real, x_fake.detach(), device)

            d_loss = (
                d_loss_real
                + d_loss_fake
                + lambda_cls * d_loss_cls
                + lambda_gp * gp
            )

            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()

            # ============================================================
            # 2. Train Generator every n_critic steps
            # ============================================================
            g_loss = torch.tensor(0.0, device=device)
            g_loss_fake = torch.tensor(0.0, device=device)
            g_loss_cls = torch.tensor(0.0, device=device)
            g_loss_rec = torch.tensor(0.0, device=device)

            if global_step % n_critic == 0:
                x_fake = G(x_real, c_trg)
                out_src_fake, out_cls_fake = D(x_fake)

                g_loss_fake = -torch.mean(out_src_fake)
                g_loss_cls = bce(out_cls_fake, c_trg)

                x_rec = G(x_fake, c_org)
                g_loss_rec = l1(x_rec, x_real)

                g_loss = (
                    g_loss_fake
                    + lambda_cls * g_loss_cls
                    + lambda_rec * g_loss_rec
                )

                g_optimizer.zero_grad(set_to_none=True)
                g_loss.backward()
                g_optimizer.step()

            pbar.set_postfix({
                "D": f"{d_loss.item():.3f}",
                "D_real": f"{d_loss_real.item():.3f}",
                "D_fake": f"{d_loss_fake.item():.3f}",
                "D_cls": f"{d_loss_cls.item():.3f}",
                "GP": f"{gp.item():.3f}",
                "G": f"{g_loss.item():.3f}",
                "G_cls": f"{g_loss_cls.item():.3f}",
                "G_rec": f"{g_loss_rec.item():.3f}",
            })

            if global_step % args.sample_interval == 0:
                sample_path = save_debug_samples(
                    G=G,
                    fixed_images=fixed_images,
                    fixed_attrs=fixed_attrs,
                    attr_names=attr_names,
                    epoch=epoch,
                    step=global_step,
                    output_dir=args.sample_dir,
                    device=device,
                )
                print(f"\nSaved samples to: {sample_path}")

        elapsed = time.time() - start_time

        print("-" * 80)
        print(f"Epoch {epoch}/{args.epochs} finished. Time: {elapsed:.2f}s")
        print("-" * 80)

        if epoch % args.save_interval == 0:
            last_path = os.path.join(args.output_dir, "last.pt")
            save_checkpoint(
                path=last_path,
                G=G,
                D=D,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                epoch=epoch,
                step=global_step,
                cfg=cfg,
                attr_names=attr_names,
            )
            print(f"Saved checkpoint to: {last_path}")

    final_path = os.path.join(args.output_dir, "final.pt")
    save_checkpoint(
        path=final_path,
        G=G,
        D=D,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        epoch=args.epochs,
        step=global_step,
        cfg=cfg,
        attr_names=attr_names,
    )

    print("=" * 80)
    print("StarGAN smoke training finished.")
    print(f"Final checkpoint: {final_path}")
    print(f"Samples saved to: {args.sample_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()