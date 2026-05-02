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
from src.data.attribute_masks import build_attribute_edit_mask, make_keep_mask
from src.models.attr_classifier import build_attr_classifier
from src.models.stargan import build_stargan_discriminator
from src.models.unet_editor import build_unet_editor


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(x):
    return (x + 1.0) / 2.0


def build_dataloader(cfg, split, batch_size, num_workers):
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


def load_attr_model(path, num_attrs, device):
    ckpt = safe_torch_load(path, map_location=device)

    model = build_attr_classifier(
        num_attrs=num_attrs,
        pretrained=False,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def gradient_penalty(D, x_real, x_fake, device):
    bsz = x_real.size(0)
    alpha = torch.rand(bsz, 1, 1, 1, device=device)

    x_hat = alpha * x_real + (1 - alpha) * x_fake
    x_hat.requires_grad_(True)

    out_src, _ = D(x_hat)
    grad_outputs = torch.ones_like(out_src)

    gradients = torch.autograd.grad(
        outputs=out_src,
        inputs=x_hat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(bsz, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    return gp


def sample_edit_plan(c_org, attr_names):
    """
    随机生成编辑计划:
        c_trg: [B, K]
        edit_mask: [B, K]

    规则:
        每个样本随机编辑 1 或 2 个属性。
        发色属性互斥处理。
    """
    device = c_org.device
    bsz, k = c_org.shape

    c_trg = c_org.clone()
    edit_mask = torch.zeros_like(c_org)

    hair_attrs = ["Blond_Hair", "Black_Hair", "Brown_Hair"]
    hair_indices = [attr_names.index(a) for a in hair_attrs if a in attr_names]

    for i in range(bsz):
        num_edit = 1 if torch.rand(1).item() < 0.7 else 2

        chosen = torch.randperm(k, device=device)[:num_edit].tolist()

        for j in chosen:
            attr = attr_names[j]

            if attr in hair_attrs and len(hair_indices) > 0:
                # 选中某种发色时，将其他发色置 0，该发色置 1
                for hj in hair_indices:
                    c_trg[i, hj] = 0.0
                    edit_mask[i, hj] = 1.0
                c_trg[i, j] = 1.0

            else:
                c_trg[i, j] = 1.0 - c_org[i, j]
                edit_mask[i, j] = 1.0

    return c_trg, edit_mask


def masked_bce_loss(logits, targets, mask):
    """
    logits: [B, K]
    targets: [B, K]
    mask: [B, K]
    """
    loss_elem = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    denom = mask.sum().clamp(min=1.0)
    return (loss_elem * mask).sum() / denom


@torch.no_grad()
def save_samples(G, fixed_images, fixed_attrs, attr_names, image_size, epoch, step, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    G.eval()

    x = fixed_images.to(device)
    c_org = fixed_attrs.to(device)

    rows = [x.cpu()]

    # 每个属性单独翻转一次
    for j, attr in enumerate(attr_names):
        c_trg = c_org.clone()
        edit_mask = torch.zeros_like(c_org)

        c_trg[:, j] = 1.0 - c_org[:, j]
        edit_mask[:, j] = 1.0

        # 发色互斥
        if attr in ["Blond_Hair", "Black_Hair", "Brown_Hair"]:
            for name in ["Blond_Hair", "Black_Hair", "Brown_Hair"]:
                if name in attr_names:
                    idx = attr_names.index(name)
                    c_trg[:, idx] = 0.0
                    edit_mask[:, idx] = 1.0
            c_trg[:, j] = 1.0

        region_mask = build_attribute_edit_mask(attr_names, edit_mask, image_size)
        x_fake, alpha, raw = G(x, c_trg, edit_mask, region_mask)

        rows.append(x_fake.cpu())

    grid = torch.cat(rows, dim=0)
    path = os.path.join(output_dir, f"epoch_{epoch:03d}_step_{step:06d}.jpg")
    save_image(denormalize(grid), path, nrow=x.size(0))

    G.train()
    return path


def save_checkpoint(path, G, D, g_opt, d_opt, epoch, step, cfg, attr_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "G_state_dict": G.state_dict(),
            "D_state_dict": D.state_dict(),
            "g_optimizer_state_dict": g_opt.state_dict(),
            "d_optimizer_state_dict": d_opt.state_dict(),
            "config": cfg,
            "attr_names": attr_names,
            "model_type": "mask_guided_unet_editor",
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/v2_mock_128_8attr.yaml")
    parser.add_argument("--attr_ckpt", type=str, default="checkpoints/attr_classifier_v2/best.pt")

    parser.add_argument("--output_dir", type=str, default="checkpoints/unet_editor_v2")
    parser.add_argument("--sample_dir", type=str, default="outputs/samples/unet_editor_v2")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--sample_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("train", {}).get("seed", 42))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    editor_cfg = cfg["v2_editor"]

    attr_names = data_cfg["selected_attrs"]
    c_dim = len(attr_names)
    image_size = data_cfg["image_size"]

    batch_size = args.batch_size or train_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else train_cfg["num_workers"]

    dataset, loader = build_dataloader(cfg, "train", batch_size, num_workers)

    attr_model = load_attr_model(args.attr_ckpt, c_dim, device)

    G = build_unet_editor(
        c_dim=c_dim,
        g_conv_dim=editor_cfg["g_conv_dim"],
        delta_scale=editor_cfg.get("delta_scale", 1.0),
    ).to(device)

    D = build_stargan_discriminator(
        image_size=image_size,
        c_dim=c_dim,
        d_conv_dim=editor_cfg["d_conv_dim"],
        d_repeat_num=editor_cfg["d_repeat_num"],
    ).to(device)

    g_opt = torch.optim.Adam(
        G.parameters(),
        lr=editor_cfg["lr"],
        betas=(editor_cfg["beta1"], editor_cfg["beta2"]),
    )

    d_opt = torch.optim.Adam(
        D.parameters(),
        lr=editor_cfg["lr"],
        betas=(editor_cfg["beta1"], editor_cfg["beta2"]),
    )

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    fixed_images, fixed_attrs, _ = next(iter(loader))
    fixed_images = fixed_images[: min(4, fixed_images.size(0))]
    fixed_attrs = fixed_attrs[: min(4, fixed_attrs.size(0))]

    print("=" * 80)
    print("Train V2 Mask-Guided U-Net Editor")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Train samples: {len(dataset)}")
    print(f"Image size: {image_size}")
    print(f"Attributes: {attr_names}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print("=" * 80)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=140)
        start = time.time()

        for batch_idx, (x_real, c_org, filenames) in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            global_step += 1

            x_real = x_real.to(device)
            c_org = c_org.to(device)

            c_trg, edit_mask = sample_edit_plan(c_org, attr_names)
            region_mask = build_attribute_edit_mask(attr_names, edit_mask, image_size)

            # =========================
            # Train D
            # =========================
            with torch.no_grad():
                x_fake, alpha, raw = G(x_real, c_trg, edit_mask, region_mask)

            out_src_real, out_cls_real = D(x_real)
            out_src_fake, _ = D(x_fake.detach())

            d_loss_real = -out_src_real.mean()
            d_loss_fake = out_src_fake.mean()
            d_loss_cls = bce(out_cls_real, c_org)

            gp = gradient_penalty(D, x_real, x_fake.detach(), device)

            d_loss = (
                d_loss_real
                + d_loss_fake
                + editor_cfg["lambda_attr"] * d_loss_cls
                + editor_cfg["lambda_gp"] * gp
            )

            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()

            # =========================
            # Train G
            # =========================
            g_loss = torch.tensor(0.0, device=device)
            g_adv = torch.tensor(0.0, device=device)
            g_attr = torch.tensor(0.0, device=device)
            g_keep = torch.tensor(0.0, device=device)
            g_rec = torch.tensor(0.0, device=device)
            g_mask = torch.tensor(0.0, device=device)
            g_alpha = torch.tensor(0.0, device=device)

            if global_step % editor_cfg["n_critic"] == 0:
                x_fake, alpha, raw = G(x_real, c_trg, edit_mask, region_mask)

                out_src_fake, _ = D(x_fake)

                g_adv = -out_src_fake.mean()

                aux_logits = attr_model(x_fake)

                keep_mask = make_keep_mask(edit_mask)

                g_attr = masked_bce_loss(aux_logits, c_trg, edit_mask)
                g_keep = masked_bce_loss(aux_logits, c_org, keep_mask)

                # cycle reconstruction
                x_rec, alpha_rec, raw_rec = G(x_fake, c_org, edit_mask, region_mask)
                g_rec = l1(x_rec, x_real)

                # outside target region should remain unchanged
                g_mask = torch.mean(torch.abs((1.0 - region_mask) * (x_fake - x_real)))

                # discourage unnecessary large editing area
                g_alpha = alpha.mean()

                g_loss = (
                    editor_cfg["lambda_adv"] * g_adv
                    + editor_cfg["lambda_attr"] * g_attr
                    + editor_cfg["lambda_keep"] * g_keep
                    + editor_cfg["lambda_rec"] * g_rec
                    + editor_cfg["lambda_mask"] * g_mask
                    + editor_cfg["lambda_alpha"] * g_alpha
                )

                g_opt.zero_grad(set_to_none=True)
                g_loss.backward()
                g_opt.step()

            pbar.set_postfix(
                {
                    "D": f"{d_loss.item():.3f}",
                    "GP": f"{gp.item():.3f}",
                    "G": f"{g_loss.item():.3f}",
                    "adv": f"{g_adv.item():.3f}",
                    "attr": f"{g_attr.item():.3f}",
                    "keep": f"{g_keep.item():.3f}",
                    "rec": f"{g_rec.item():.3f}",
                    "mask": f"{g_mask.item():.3f}",
                    "alpha": f"{g_alpha.item():.3f}",
                }
            )

            if global_step % args.sample_interval == 0:
                path = save_samples(
                    G,
                    fixed_images,
                    fixed_attrs,
                    attr_names,
                    image_size,
                    epoch,
                    global_step,
                    args.sample_dir,
                    device,
                )
                print(f"\nSaved sample: {path}")

        elapsed = time.time() - start
        print(f"Epoch {epoch} finished. Time: {elapsed:.2f}s")

        if epoch % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, "last.pt"),
                G,
                D,
                g_opt,
                d_opt,
                epoch,
                global_step,
                cfg,
                attr_names,
            )

    save_checkpoint(
        os.path.join(args.output_dir, "final.pt"),
        G,
        D,
        g_opt,
        d_opt,
        args.epochs,
        global_step,
        cfg,
        attr_names,
    )

    print("=" * 80)
    print("V2 editor training finished.")
    print(f"Final checkpoint: {os.path.join(args.output_dir, 'final.pt')}")
    print("=" * 80)


if __name__ == "__main__":
    main()