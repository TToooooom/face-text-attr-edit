import os
from typing import List, Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image


class CelebAAttrDataset(Dataset):
    """
    CelebA 属性数据集读取器。

    功能：
    1. 读取 img_align_celeba 中的人脸图像；
    2. 读取 list_attr_celeba.txt 中的 40 个属性；
    3. 选择 MVP 阶段需要的属性，例如 Smiling / Eyeglasses；
    4. 根据 list_eval_partition.txt 划分 train / valid / test；
    5. 将 CelebA 原始标签 -1/1 转为 0/1。

    返回：
        image: Tensor, shape = [3, H, W]
        attrs: Tensor, shape = [K], K 是属性数量
        filename: str
    """

    SPLIT_TO_ID = {
        "train": 0,
        "valid": 1,
        "val": 1,
        "test": 2,
        "all": -1,
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        selected_attrs: Optional[List[str]] = None,
        transform=None,
        image_dir: str = "img_align_celeba",
        attr_file: str = "list_attr_celeba.txt",
        partition_file: str = "list_eval_partition.txt",
    ):
        super().__init__()

        if selected_attrs is None:
            selected_attrs = ["Smiling", "Eyeglasses"]

        if split not in self.SPLIT_TO_ID:
            raise ValueError(f"Unsupported split: {split}. "
                             f"Expected one of {list(self.SPLIT_TO_ID.keys())}")

        self.root = root
        self.split = split
        self.selected_attrs = selected_attrs
        self.transform = transform

        self.image_dir = os.path.join(root, image_dir)
        self.attr_path = os.path.join(root, attr_file)
        self.partition_path = os.path.join(root, partition_file)

        self._check_files()

        self.attr_names, self.attr_table = self._load_attr_file(self.attr_path)
        self.partition_table = self._load_partition_file(self.partition_path)

        self.selected_attr_indices = self._get_selected_attr_indices(
            self.attr_names,
            self.selected_attrs
        )

        self.samples = self._build_samples()

    def _check_files(self):
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}\n"
                f"Expected structure: data/celeba/img_align_celeba/000001.jpg ..."
            )

        if not os.path.isfile(self.attr_path):
            raise FileNotFoundError(f"Attribute file not found: {self.attr_path}")

        if not os.path.isfile(self.partition_path):
            raise FileNotFoundError(f"Partition file not found: {self.partition_path}")

    @staticmethod
    def _load_attr_file(attr_path: str) -> Tuple[List[str], Dict[str, List[int]]]:
        """
        读取 list_attr_celeba.txt。

        文件格式通常为：
            第一行: 图像数量
            第二行: 40 个属性名
            后续每行: image_name attr_1 attr_2 ... attr_40

        原始属性取值为 -1 / 1。
        """
        with open(attr_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        attr_names = lines[1].strip().split()
        attr_table = {}

        for line in lines[2:]:
            parts = line.strip().split()
            filename = parts[0]
            attrs = [int(x) for x in parts[1:]]
            attr_table[filename] = attrs

        return attr_names, attr_table

    @staticmethod
    def _load_partition_file(partition_path: str) -> Dict[str, int]:
        """
        读取 list_eval_partition.txt。

        split id:
            0: train
            1: valid
            2: test
        """
        partition_table = {}

        with open(partition_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                filename, split_id = parts
                partition_table[filename] = int(split_id)

        return partition_table

    @staticmethod
    def _get_selected_attr_indices(
        attr_names: List[str],
        selected_attrs: List[str]
    ) -> List[int]:
        indices = []

        for attr in selected_attrs:
            if attr not in attr_names:
                raise ValueError(
                    f"Attribute {attr} not found in CelebA attributes.\n"
                    f"Available attributes include: {attr_names}"
                )
            indices.append(attr_names.index(attr))

        return indices

    def _build_samples(self):
        target_split_id = self.SPLIT_TO_ID[self.split]
        samples = []

        for filename, attrs in self.attr_table.items():
            if filename not in self.partition_table:
                continue

            split_id = self.partition_table[filename]

            if target_split_id != -1 and split_id != target_split_id:
                continue

            image_path = os.path.join(self.image_dir, filename)
            if not os.path.isfile(image_path):
                continue

            selected_values = []
            for idx in self.selected_attr_indices:
                # CelebA 原始标签：-1 表示无属性，1 表示有属性
                # 训练时改成 0 / 1
                value = 1 if attrs[idx] == 1 else 0
                selected_values.append(value)

            samples.append((image_path, selected_values, filename))

        if len(samples) == 0:
            raise RuntimeError(
                f"No samples found for split={self.split}. "
                f"Please check image directory and annotation files."
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, attr_values, filename = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        attrs = torch.tensor(attr_values, dtype=torch.float32)

        return image, attrs, filename

    def get_attr_names(self) -> List[str]:
        return self.selected_attrs

    def get_num_attrs(self) -> int:
        return len(self.selected_attrs)