import os
import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.data.celeba_dataset import CelebAAttrDataset
from src.data.transforms import build_celeba_transform


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """
    将 [-1, 1] 的图像张量还原到 [0, 1]，用于保存可视化。
    """
    return (x + 1.0) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mvp_64.yaml")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "valid", "val", "test", "all"])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    celeba_root = cfg["data"]["celeba_root"]
    image_size = cfg["data"]["image_size"]
    crop_size = cfg["data"]["crop_size"]
    selected_attrs = cfg["data"]["selected_attrs"]

    transform = build_celeba_transform(
        image_size=image_size,
        crop_size=crop_size,
        mode="eval"
    )

    dataset = CelebAAttrDataset(
        root=celeba_root,
        split=args.split,
        selected_attrs=selected_attrs,
        transform=transform,
        image_dir=cfg["data"]["image_dir"],
        attr_file=cfg["data"]["attr_file"],
        partition_file=cfg["data"]["partition_file"],
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    print("=" * 80)
    print("CelebA Dataset Check")
    print("=" * 80)
    print(f"Root: {celeba_root}")
    print(f"Split: {args.split}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Selected attributes: {dataset.get_attr_names()}")
    print(f"Number of attributes: {dataset.get_num_attrs()}")

    images, attrs, filenames = next(iter(loader))

    print("-" * 80)
    print(f"Image batch shape: {images.shape}")
    print(f"Attribute batch shape: {attrs.shape}")
    print(f"Image tensor range: min={images.min().item():.4f}, max={images.max().item():.4f}")
    print("First few filenames:")
    for name in filenames[:5]:
        print("  ", name)

    print("-" * 80)
    print("First few attribute labels:")
    for i in range(min(5, attrs.size(0))):
        label_dict = {
            attr_name: int(attrs[i, j].item())
            for j, attr_name in enumerate(dataset.get_attr_names())
        }
        print(f"  {filenames[i]}: {label_dict}")

    os.makedirs("outputs/samples", exist_ok=True)
    save_path = f"outputs/samples/check_celeba_{args.split}.jpg"

    save_image(
        denormalize(images[:args.batch_size]),
        save_path,
        nrow=4
    )

    print("-" * 80)
    print(f"Saved sample grid to: {save_path}")
    print("Dataset check finished successfully.")


if __name__ == "__main__":
    main()