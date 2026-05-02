import os
import sys
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
from src.models.attr_classifier import build_attr_classifier
from src.models.stargan import build_stargan_generator


def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(cfg, split: str, batch_size: int, num_workers: int):
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


def load_attr_classifier(ckpt_path: str, num_attrs: int, device):
    ckpt = safe_torch_load(ckpt_path, map_location=device)

    model = build_attr_classifier(
        num_attrs=num_attrs,
        pretrained=False,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    attr_names = ckpt.get("attr_names", None)

    return model, attr_names


def load_stargan_generator(ckpt_path: str, device):
    ckpt = safe_torch_load(ckpt_path, map_location=device)

    cfg = ckpt["config"]
    attr_names = ckpt["attr_names"]
    stargan_cfg = cfg["stargan"]

    G = build_stargan_generator(
        c_dim=len(attr_names),
        g_conv_dim=stargan_cfg["g_conv_dim"],
        g_repeat_num=stargan_cfg["g_repeat_num"],
    ).to(device)

    G.load_state_dict(ckpt["G_state_dict"], strict=True)
    G.eval()

    return G, attr_names


def sample_target_labels(c_org: torch.Tensor, mode: str = "random"):
    """
    构造目标属性 c_trg。

    c_org:
        [B, K], 0/1

    mode:
        random:
            随机采样目标属性，如果和原属性完全相同，则随机翻转一个属性。

        flip:
            所有属性全部翻转。MVP 下就是:
                [Smiling, Eyeglasses] -> [1-Smiling, 1-Eyeglasses]
    """
    if mode == "flip":
        return 1.0 - c_org

    if mode != "random":
        raise ValueError(f"Unsupported target mode: {mode}")

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


@torch.no_grad()
def predict_attrs(attr_model, images, threshold: float = 0.5):
    """
    用属性分类器预测编辑后图像属性。

    返回:
        pred: [B, K], 0/1
        prob: [B, K]
    """
    logits = attr_model(images)
    prob = torch.sigmoid(logits)
    pred = (prob >= threshold).float()
    return pred, prob


def save_eval_samples(
    x_real,
    x_fake,
    x_rec,
    c_org,
    c_trg,
    attr_pred,
    filenames,
    attr_names,
    output_dir,
    batch_idx,
):
    """
    保存可视化结果。

    图像排列：
        第一行：原图 x
        第二行：编辑图 G(x, c_trg)
        第三行：重建图 G(G(x,c_trg), c_org)
    """
    os.makedirs(output_dir, exist_ok=True)

    bsz = x_real.size(0)

    grid = torch.cat(
        [
            x_real.cpu(),
            x_fake.cpu(),
            x_rec.cpu(),
        ],
        dim=0,
    )

    image_path = os.path.join(output_dir, f"eval_batch_{batch_idx:04d}.jpg")
    save_image(denormalize(grid), image_path, nrow=bsz)

    info_path = os.path.join(output_dir, f"eval_batch_{batch_idx:04d}.txt")

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Each column corresponds to one image.\n")
        f.write("Rows: original | edited | reconstructed\n\n")

        for i in range(bsz):
            f.write(f"Image {i}: {filenames[i]}\n")

            org_dict = {
                name: int(c_org[i, j].item())
                for j, name in enumerate(attr_names)
            }

            trg_dict = {
                name: int(c_trg[i, j].item())
                for j, name in enumerate(attr_names)
            }

            pred_dict = {
                name: int(attr_pred[i, j].item())
                for j, name in enumerate(attr_names)
            }

            f.write(f"  original attrs: {org_dict}\n")
            f.write(f"  target attrs:   {trg_dict}\n")
            f.write(f"  pred attrs:     {pred_dict}\n")
            f.write("\n")

    return image_path, info_path


@torch.no_grad()
def evaluate(
    G,
    attr_model,
    loader,
    attr_names,
    device,
    attr_threshold: float,
    target_mode: str,
    output_dir: str,
    max_batches: int | None,
    save_samples: int,
):
    l1 = nn.L1Loss(reduction="none")

    num_attrs = len(attr_names)

    total_samples = 0

    # 所有属性位准确率
    attr_correct_sum = torch.zeros(num_attrs)
    attr_count_sum = torch.zeros(num_attrs)

    # 只统计被改变的属性位
    changed_correct_sum = torch.zeros(num_attrs)
    changed_count_sum = torch.zeros(num_attrs)

    exact_match_correct = 0

    # 重建误差
    rec_l1_sum = 0.0

    pbar = tqdm(loader, desc="Evaluate", ncols=120)

    for batch_idx, (x_real, c_org, filenames) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x_real = x_real.to(device, non_blocking=True)
        c_org = c_org.to(device, non_blocking=True)

        c_trg = sample_target_labels(c_org, mode=target_mode)

        # 1. 编辑
        x_fake = G(x_real, c_trg)

        # 2. 重建
        x_rec = G(x_fake, c_org)

        # 3. 用属性分类器判断编辑后属性
        attr_pred, attr_prob = predict_attrs(
            attr_model=attr_model,
            images=x_fake,
            threshold=attr_threshold,
        )

        # 4. 属性编辑准确率
        correct = (attr_pred == c_trg).float()

        attr_correct_sum += correct.detach().cpu().sum(dim=0)
        attr_count_sum += torch.ones_like(correct).detach().cpu().sum(dim=0)

        # 5. 只看发生改变的属性
        changed_mask = (c_trg != c_org).float()
        changed_correct = correct * changed_mask

        changed_correct_sum += changed_correct.detach().cpu().sum(dim=0)
        changed_count_sum += changed_mask.detach().cpu().sum(dim=0)

        # 6. exact match：所有属性都等于目标属性
        exact_match_correct += correct.all(dim=1).float().sum().item()

        # 7. 重建 L1
        per_sample_l1 = l1(x_rec, x_real).mean(dim=[1, 2, 3])
        rec_l1_sum += per_sample_l1.sum().item()

        bsz = x_real.size(0)
        total_samples += bsz

        # 8. 保存可视化
        if batch_idx < save_samples:
            sample_img_path, sample_info_path = save_eval_samples(
                x_real=x_real,
                x_fake=x_fake,
                x_rec=x_rec,
                c_org=c_org,
                c_trg=c_trg,
                attr_pred=attr_pred,
                filenames=filenames,
                attr_names=attr_names,
                output_dir=output_dir,
                batch_idx=batch_idx,
            )

        overall_attr_acc = attr_correct_sum.sum().item() / max(attr_count_sum.sum().item(), 1.0)
        exact_match_acc = exact_match_correct / max(total_samples, 1)
        avg_rec_l1 = rec_l1_sum / max(total_samples, 1)

        pbar.set_postfix({
            "attr_acc": f"{overall_attr_acc:.4f}",
            "exact": f"{exact_match_acc:.4f}",
            "rec_l1": f"{avg_rec_l1:.4f}",
        })

    per_attr_acc = attr_correct_sum / torch.clamp(attr_count_sum, min=1.0)

    changed_per_attr_acc = changed_correct_sum / torch.clamp(changed_count_sum, min=1.0)

    overall_attr_acc = attr_correct_sum.sum().item() / max(attr_count_sum.sum().item(), 1.0)

    if changed_count_sum.sum().item() > 0:
        overall_changed_acc = changed_correct_sum.sum().item() / changed_count_sum.sum().item()
    else:
        overall_changed_acc = 0.0

    exact_match_acc = exact_match_correct / max(total_samples, 1)
    avg_rec_l1 = rec_l1_sum / max(total_samples, 1)

    results = {
        "num_samples": total_samples,
        "overall_attr_acc": overall_attr_acc,
        "overall_changed_attr_acc": overall_changed_acc,
        "exact_match_acc": exact_match_acc,
        "avg_rec_l1": avg_rec_l1,
        "per_attr_acc": {
            name: float(per_attr_acc[i].item())
            for i, name in enumerate(attr_names)
        },
        "changed_per_attr_acc": {
            name: float(changed_per_attr_acc[i].item())
            for i, name in enumerate(attr_names)
        },
        "changed_count": {
            name: int(changed_count_sum[i].item())
            for i, name in enumerate(attr_names)
        },
    }

    return results


def save_results(results: dict, output_dir: str):
    import json

    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "metrics.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return path


def print_results(results: dict):
    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    print(f"Num samples: {results['num_samples']}")
    print(f"Overall attribute accuracy:        {results['overall_attr_acc']:.4f}")
    print(f"Changed attribute accuracy:        {results['overall_changed_attr_acc']:.4f}")
    print(f"Exact match accuracy:              {results['exact_match_acc']:.4f}")
    print(f"Average reconstruction L1:         {results['avg_rec_l1']:.6f}")

    print("-" * 80)
    print("Per-attribute accuracy:")
    for name, value in results["per_attr_acc"].items():
        print(f"  {name:12s}: {value:.4f}")

    print("-" * 80)
    print("Changed per-attribute accuracy:")
    for name, value in results["changed_per_attr_acc"].items():
        count = results["changed_count"][name]
        print(f"  {name:12s}: {value:.4f}  | changed count = {count}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/mvp_64.yaml")

    parser.add_argument("--attr_ckpt", type=str, default="checkpoints/attr_classifier/best.pt")
    parser.add_argument("--stargan_ckpt", type=str, default="checkpoints/stargan/final.pt")

    parser.add_argument("--output_dir", type=str, default="outputs/eval")

    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "val", "test", "all"])

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--attr_threshold", type=float, default=0.5)

    parser.add_argument("--target_mode", type=str, default="random", choices=["random", "flip"])

    # 本地 mock 测试用
    parser.add_argument("--max_batches", type=int, default=None)

    # 保存前几个 batch 的可视化结果
    parser.add_argument("--save_samples", type=int, default=3)

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

    attr_names = data_cfg["selected_attrs"]
    num_attrs = len(attr_names)

    batch_size = args.batch_size if args.batch_size is not None else train_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else train_cfg["num_workers"]

    dataset, loader = build_dataloader(
        cfg=cfg,
        split=args.split,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    attr_model, attr_ckpt_attrs = load_attr_classifier(
        ckpt_path=args.attr_ckpt,
        num_attrs=num_attrs,
        device=device,
    )

    G, stargan_ckpt_attrs = load_stargan_generator(
        ckpt_path=args.stargan_ckpt,
        device=device,
    )

    if attr_ckpt_attrs is not None and attr_ckpt_attrs != attr_names:
        print(f"[Warning] attr classifier attrs {attr_ckpt_attrs} != config attrs {attr_names}")

    if stargan_ckpt_attrs != attr_names:
        print(f"[Warning] stargan attrs {stargan_ckpt_attrs} != config attrs {attr_names}")

    print("=" * 80)
    print("Phase 5: Evaluate StarGAN")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Num samples in dataset: {len(dataset)}")
    print(f"Attributes: {attr_names}")
    print(f"Batch size: {batch_size}")
    print(f"Target mode: {args.target_mode}")
    print(f"Attr threshold: {args.attr_threshold}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    results = evaluate(
        G=G,
        attr_model=attr_model,
        loader=loader,
        attr_names=attr_names,
        device=device,
        attr_threshold=args.attr_threshold,
        target_mode=args.target_mode,
        output_dir=args.output_dir,
        max_batches=args.max_batches,
        save_samples=args.save_samples,
    )

    print_results(results)

    metrics_path = save_results(results, args.output_dir)
    print(f"Saved metrics to: {metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()