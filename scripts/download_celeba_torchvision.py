import argparse

from torchvision.datasets import CelebA
from src.data.transforms import build_celeba_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--crop_size", type=int, default=178)
    args = parser.parse_args()

    transform = build_celeba_transform(
        image_size=args.image_size,
        crop_size=args.crop_size,
        mode="eval",
    )

    print("=" * 80)
    print("Downloading CelebA by torchvision")
    print("=" * 80)
    print(f"Root: {args.root}")
    print("Expected final directory: data/celeba/")
    print("This may take a long time because CelebA has more than 200k images.")

    # 关键点：
    # root="data" 时，torchvision 会把 CelebA 放到 data/celeba/
    dataset = CelebA(
        root=args.root,
        split="train",
        target_type="attr",
        transform=transform,
        download=True,
    )

    print("=" * 80)
    print("Download finished and dataset loaded.")
    print(f"Train split size: {len(dataset)}")

    image, attr = dataset[0]
    print(f"One image shape: {image.shape}")
    print(f"One attr shape: {attr.shape}")
    print(f"Attr dtype: {attr.dtype}")
    print(f"Attr min/max: {attr.min().item()} / {attr.max().item()}")
    print("=" * 80)


if __name__ == "__main__":
    main()