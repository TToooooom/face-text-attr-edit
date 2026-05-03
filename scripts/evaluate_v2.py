import os
import sys
import json
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.celeba_dataset import CelebAAttrDataset
from src.data.transforms import build_celeba_transform
from src.data.attribute_masks import build_attribute_edit_mask
from src.models.attr_classifier import build_attr_classifier
from src.models.unet_editor import build_unet_editor


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def denormalize(x):
    return (x + 1.0) / 2.0


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    loss_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return loss_h + loss_w


def build_dataloader(cfg, split, batch_size, num_workers):
    data_cfg = cfg["data"]

    transform = build_celeba_transform(
        image_size=data_cfg["image_size"],
        crop_size=data_cfg["crop_size"],
        mode="eval",
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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataset, loader


def load_attr_classifier(path, num_attrs, device):
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


def load_editor(path, device):
    ckpt = safe_torch_load(path, map_location=device)

    cfg = ckpt["config"]
    attr_names = ckpt["attr_names"]
    editor_cfg = cfg["v2_editor"]

    G = build_unet_editor(
        c_dim=len(attr_names),
        g_conv_dim=editor_cfg["g_conv_dim"],
        delta_scale=editor_cfg.get("delta_scale", 1.0),
    ).to(device)

    G.load_state_dict(ckpt["G_state_dict"], strict=True)
    G.eval()

    return G, cfg, attr_names


def sample_edit_plan(c_org, attr_names, mode="random", attr_name=None, max_edit_attrs=2):
    """
    生成目标属性向量 c_trg 和 edit_mask。

    mode:
        random: 每个样本随机编辑 1~max_edit_attrs 个属性
        attr  : 所有样本只编辑 attr_name 指定属性
    """
    device = c_org.device
    bsz, k = c_org.shape

    c_trg = c_org.clone()
    edit_mask = torch.zeros_like(c_org)

    hair_attrs = ["Blond_Hair", "Black_Hair", "Brown_Hair"]
    hair_indices = [attr_names.index(a) for a in hair_attrs if a in attr_names]

    def edit_one(i, j):
        attr = attr_names[j]

        if attr in hair_attrs and len(hair_indices) > 0:
            # 发色互斥：选择一种发色时，其他发色置 0
            for hj in hair_indices:
                c_trg[i, hj] = 0.0
                edit_mask[i, hj] = 1.0
            c_trg[i, j] = 1.0
        else:
            c_trg[i, j] = 1.0 - c_org[i, j]
            edit_mask[i, j] = 1.0

    if mode == "attr":
        if attr_name is None:
            raise ValueError("--attr_name must be provided when --target_mode attr.")
        if attr_name not in attr_names:
            raise ValueError(f"Unknown attr_name={attr_name}. Valid attrs: {attr_names}")

        j = attr_names.index(attr_name)
        for i in range(bsz):
            edit_one(i, j)

    elif mode == "random":
        max_edit_attrs = max(1, min(max_edit_attrs, k))

        for i in range(bsz):
            if max_edit_attrs == 1:
                num_edit = 1
            else:
                # 70% 单属性，30% 多属性
                num_edit = 1 if torch.rand(1).item() < 0.7 else max_edit_attrs

            chosen = torch.randperm(k, device=device)[:num_edit].tolist()
            for j in chosen:
                edit_one(i, j)

    else:
        raise ValueError(f"Unknown target_mode: {mode}")

    return c_trg, edit_mask


def masked_accuracy(pred_binary, target_binary, mask):
    """
    pred_binary/target_binary/mask: [B, K]
    返回 correct, total
    """
    correct = ((pred_binary == target_binary) * mask.bool()).sum().item()
    total = mask.sum().item()
    return correct, total


def masked_l1_mean(x, y, mask):
    """
    x, y: [B, 3, H, W]
    mask: [B, 1, H, W], 1 表示参与计算区域
    """
    diff = torch.abs(x - y) * mask
    denom = (mask.sum() * x.size(1)).clamp(min=1.0)
    return diff.sum() / denom


def save_visual_batch(
    x_real,
    x_fake,
    alpha,
    region_mask,
    filenames,
    output_dir,
    batch_index,
    max_images=8,
):
    os.makedirs(output_dir, exist_ok=True)

    n = min(max_images, x_real.size(0))

    x_vis = denormalize(x_real[:n].cpu())
    fake_vis = denormalize(x_fake[:n].cpu())

    diff_vis = torch.abs(x_fake[:n].cpu() - x_real[:n].cpu()) / 2.0
    diff_vis = diff_vis.clamp(0.0, 1.0)

    alpha_vis = alpha[:n].cpu().repeat(1, 3, 1, 1)
    mask_vis = region_mask[:n].cpu().repeat(1, 3, 1, 1)

    grid = torch.cat(
        [
            x_vis,
            fake_vis,
            diff_vis,
            alpha_vis,
            mask_vis,
        ],
        dim=0,
    )

    save_path = os.path.join(output_dir, f"batch_{batch_index:04d}_grid.jpg")
    save_image(grid, save_path, nrow=n)

    return save_path


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/v2_server_128_8attr.yaml")
    parser.add_argument("--attr_ckpt", type=str, default="checkpoints/attr_classifier_v2/best.pt")
    parser.add_argument("--editor_ckpt", type=str, default="checkpoints/unet_editor_v2/final.pt")

    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--target_mode", type=str, default="random", choices=["random", "attr"])
    parser.add_argument("--attr_name", type=str, default=None)
    parser.add_argument("--max_edit_attrs", type=int, default=2)

    parser.add_argument("--attr_threshold", type=float, default=0.5)
    parser.add_argument("--max_batches", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default="outputs/eval_v2")
    parser.add_argument("--save_samples", type=int, default=5)
    parser.add_argument("--sample_max_images", type=int, default=8)

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    attr_names = data_cfg["selected_attrs"]
    c_dim = len(attr_names)
    image_size = data_cfg["image_size"]

    os.makedirs(args.output_dir, exist_ok=True)

    dataset, loader = build_dataloader(
        cfg=cfg,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    attr_model = load_attr_classifier(args.attr_ckpt, c_dim, device)
    G, ckpt_cfg, ckpt_attrs = load_editor(args.editor_ckpt, device)

    if ckpt_attrs != attr_names:
        raise ValueError(
            f"Attribute mismatch between config and editor checkpoint:\n"
            f"config attrs = {attr_names}\n"
            f"ckpt attrs   = {ckpt_attrs}"
        )

    print("=" * 80)
    print("Evaluate V2 Mask-Guided U-Net Editor")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Num samples in dataset: {len(dataset)}")
    print(f"Attributes: {attr_names}")
    print(f"Batch size: {args.batch_size}")
    print(f"Target mode: {args.target_mode}")
    print(f"Attr threshold: {args.attr_threshold}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    total_samples = 0
    total_attr_correct = 0
    total_attr_count = 0
    exact_match_count = 0

    changed_correct = 0
    changed_total = 0

    keep_correct = 0
    keep_total = 0

    per_attr_correct = torch.zeros(c_dim, dtype=torch.float64)
    per_attr_total = torch.zeros(c_dim, dtype=torch.float64)

    changed_per_attr_correct = torch.zeros(c_dim, dtype=torch.float64)
    changed_per_attr_total = torch.zeros(c_dim, dtype=torch.float64)

    keep_per_attr_correct = torch.zeros(c_dim, dtype=torch.float64)
    keep_per_attr_total = torch.zeros(c_dim, dtype=torch.float64)

    total_l1_sum = 0.0
    outside_l1_sum = 0.0
    inside_l1_sum = 0.0
    self_l1_sum = 0.0
    rec_l1_sum = 0.0
    alpha_sum = 0.0
    tv_sum = 0.0

    saved_batches = 0

    pbar = tqdm(loader, desc="Evaluate V2", ncols=140)

    for batch_idx, (x_real, c_org, filenames) in enumerate(pbar):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        x_real = x_real.to(device)
        c_org = c_org.to(device)

        bsz = x_real.size(0)

        c_trg, edit_mask = sample_edit_plan(
            c_org=c_org,
            attr_names=attr_names,
            mode=args.target_mode,
            attr_name=args.attr_name,
            max_edit_attrs=args.max_edit_attrs,
        )

        region_mask = build_attribute_edit_mask(
            attr_names=attr_names,
            edit_mask=edit_mask,
            image_size=image_size,
        )

        x_fake, alpha, raw = G(x_real, c_trg, edit_mask, region_mask)

        # 属性预测
        logits_fake = attr_model(x_fake)
        probs_fake = torch.sigmoid(logits_fake)
        pred_fake = (probs_fake >= args.attr_threshold).float()

        target_binary = c_trg.float()

        # 整体属性准确率
        attr_correct_tensor = (pred_fake == target_binary).float()
        total_attr_correct += attr_correct_tensor.sum().item()
        total_attr_count += bsz * c_dim

        exact_match_count += attr_correct_tensor.all(dim=1).sum().item()

        # changed / keep 属性准确率
        cc, ct = masked_accuracy(pred_fake, target_binary, edit_mask)
        changed_correct += cc
        changed_total += ct

        keep_mask = 1.0 - edit_mask
        kc, kt = masked_accuracy(pred_fake, c_org, keep_mask)
        keep_correct += kc
        keep_total += kt

        # per-attribute
        for j in range(c_dim):
            per_attr_correct[j] += attr_correct_tensor[:, j].sum().item()
            per_attr_total[j] += bsz

            changed_mask_j = edit_mask[:, j]
            if changed_mask_j.sum().item() > 0:
                changed_per_attr_correct[j] += (
                    ((pred_fake[:, j] == target_binary[:, j]) * changed_mask_j.bool()).sum().item()
                )
                changed_per_attr_total[j] += changed_mask_j.sum().item()

            keep_mask_j = keep_mask[:, j]
            if keep_mask_j.sum().item() > 0:
                keep_per_attr_correct[j] += (
                    ((pred_fake[:, j] == c_org[:, j]) * keep_mask_j.bool()).sum().item()
                )
                keep_per_attr_total[j] += keep_mask_j.sum().item()

        # 图像变化指标
        total_l1 = torch.mean(torch.abs(x_fake - x_real))
        outside_l1 = masked_l1_mean(x_fake, x_real, 1.0 - region_mask)
        inside_l1 = masked_l1_mean(x_fake, x_real, region_mask)

        # self reconstruction
        zero_edit_mask = torch.zeros_like(edit_mask)
        full_region_mask = torch.ones_like(region_mask)

        x_self, alpha_self, raw_self = G(
            x_real,
            c_org,
            zero_edit_mask,
            full_region_mask,
        )
        self_l1 = torch.mean(torch.abs(x_self - x_real))

        # cycle reconstruction
        x_rec, alpha_rec, raw_rec = G(
            x_fake,
            c_org,
            edit_mask,
            region_mask,
        )
        rec_l1 = torch.mean(torch.abs(x_rec - x_real))

        alpha_mean = alpha.mean()
        tv_alpha = tv_loss(alpha)

        total_l1_sum += total_l1.item() * bsz
        outside_l1_sum += outside_l1.item() * bsz
        inside_l1_sum += inside_l1.item() * bsz
        self_l1_sum += self_l1.item() * bsz
        rec_l1_sum += rec_l1.item() * bsz
        alpha_sum += alpha_mean.item() * bsz
        tv_sum += tv_alpha.item() * bsz

        total_samples += bsz

        if saved_batches < args.save_samples:
            vis_dir = os.path.join(args.output_dir, "samples")
            save_visual_batch(
                x_real=x_real,
                x_fake=x_fake,
                alpha=alpha,
                region_mask=region_mask,
                filenames=filenames,
                output_dir=vis_dir,
                batch_index=batch_idx,
                max_images=args.sample_max_images,
            )
            saved_batches += 1

        overall_acc = total_attr_correct / max(total_attr_count, 1)
        changed_acc = changed_correct / max(changed_total, 1)
        keep_acc = keep_correct / max(keep_total, 1)

        pbar.set_postfix(
            {
                "overall": f"{overall_acc:.4f}",
                "changed": f"{changed_acc:.4f}",
                "keep": f"{keep_acc:.4f}",
                "out_l1": f"{outside_l1_sum / max(total_samples, 1):.4f}",
                "alpha": f"{alpha_sum / max(total_samples, 1):.4f}",
            }
        )

    # 汇总
    metrics = {}

    metrics["num_samples"] = int(total_samples)
    metrics["overall_attribute_accuracy"] = total_attr_correct / max(total_attr_count, 1)
    metrics["changed_attribute_accuracy"] = changed_correct / max(changed_total, 1)
    metrics["keep_attribute_accuracy"] = keep_correct / max(keep_total, 1)
    metrics["exact_match_accuracy"] = exact_match_count / max(total_samples, 1)

    metrics["total_l1"] = total_l1_sum / max(total_samples, 1)
    metrics["outside_mask_l1"] = outside_l1_sum / max(total_samples, 1)
    metrics["inside_mask_l1"] = inside_l1_sum / max(total_samples, 1)
    metrics["self_l1"] = self_l1_sum / max(total_samples, 1)
    metrics["rec_l1"] = rec_l1_sum / max(total_samples, 1)
    metrics["average_alpha"] = alpha_sum / max(total_samples, 1)
    metrics["tv_alpha"] = tv_sum / max(total_samples, 1)

    metrics["per_attribute_accuracy"] = {}
    metrics["changed_per_attribute_accuracy"] = {}
    metrics["keep_per_attribute_accuracy"] = {}

    for j, attr in enumerate(attr_names):
        metrics["per_attribute_accuracy"][attr] = (
            float(per_attr_correct[j] / per_attr_total[j])
            if per_attr_total[j] > 0
            else None
        )

        metrics["changed_per_attribute_accuracy"][attr] = {
            "accuracy": (
                float(changed_per_attr_correct[j] / changed_per_attr_total[j])
                if changed_per_attr_total[j] > 0
                else None
            ),
            "count": int(changed_per_attr_total[j].item()),
        }

        metrics["keep_per_attribute_accuracy"][attr] = {
            "accuracy": (
                float(keep_per_attr_correct[j] / keep_per_attr_total[j])
                if keep_per_attr_total[j] > 0
                else None
            ),
            "count": int(keep_per_attr_total[j].item()),
        }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("V2 Evaluation Results")
    print("=" * 80)
    print(f"Num samples: {metrics['num_samples']}")
    print(f"Overall attribute accuracy:        {metrics['overall_attribute_accuracy']:.4f}")
    print(f"Changed attribute accuracy:        {metrics['changed_attribute_accuracy']:.4f}")
    print(f"Keep attribute accuracy:           {metrics['keep_attribute_accuracy']:.4f}")
    print(f"Exact match accuracy:              {metrics['exact_match_accuracy']:.4f}")
    print("-" * 80)
    print(f"Total L1:                          {metrics['total_l1']:.6f}")
    print(f"Outside-mask L1:                   {metrics['outside_mask_l1']:.6f}")
    print(f"Inside-mask L1:                    {metrics['inside_mask_l1']:.6f}")
    print(f"Self reconstruction L1:            {metrics['self_l1']:.6f}")
    print(f"Cycle reconstruction L1:           {metrics['rec_l1']:.6f}")
    print(f"Average alpha:                     {metrics['average_alpha']:.6f}")
    print(f"TV(alpha):                         {metrics['tv_alpha']:.6f}")
    print("-" * 80)
    print("Per-attribute accuracy:")
    for attr, acc in metrics["per_attribute_accuracy"].items():
        print(f"  {attr:12s}: {acc:.4f}")
    print("-" * 80)
    print("Changed per-attribute accuracy:")
    for attr, item in metrics["changed_per_attribute_accuracy"].items():
        acc = item["accuracy"]
        count = item["count"]
        if acc is None:
            print(f"  {attr:12s}: None    | changed count = {count}")
        else:
            print(f"  {attr:12s}: {acc:.4f}  | changed count = {count}")
    print("-" * 80)
    print("Keep per-attribute accuracy:")
    for attr, item in metrics["keep_per_attribute_accuracy"].items():
        acc = item["accuracy"]
        count = item["count"]
        if acc is None:
            print(f"  {attr:12s}: None    | keep count = {count}")
        else:
            print(f"  {attr:12s}: {acc:.4f}  | keep count = {count}")

    print("=" * 80)
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved samples to: {os.path.join(args.output_dir, 'samples')}")
    print("=" * 80)


if __name__ == "__main__":
    main()