from torchvision import transforms


def build_celeba_transform(image_size: int = 64, crop_size: int = 178, mode: str = "train"):
    """
    CelebA 图像预处理。

    StarGAN 训练时一般将图像归一化到 [-1, 1]，
    因为生成器最后常用 tanh 输出。
    """
    if mode == "train":
        return transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])