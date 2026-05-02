import re
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


KEEP = 0
SET0 = 1
SET1 = 2

ACTION_TO_ID = {
    "KEEP": KEEP,
    "SET0": SET0,
    "SET1": SET1,
}

ID_TO_ACTION = {
    KEEP: "KEEP",
    SET0: "SET0",
    SET1: "SET1",
}


SPECIAL_TOKENS = {
    "PAD": "<pad>",
    "UNK": "<unk>",
    "CLS": "<cls>",
}


def normalize_text(text: str) -> str:
    """
    简单英文文本清洗：
    1. 小写；
    2. 去掉多余标点；
    3. 压缩空格。
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    if len(text) == 0:
        return []
    return text.split()


def get_attr_phrase_templates() -> Dict[str, Dict[int, List[str]]]:
    """
    每个属性对应 SET0 / SET1 的文本模板。

    注意：
    KEEP 不直接写模板。
    当一个指令没有提到某属性时，该属性自动视为 KEEP。
    """
    templates = {
        "Smiling": {
            SET1: [
                "make the person smile",
                "add a smile",
                "make the face smiling",
                "make the person look happy",
                "make the face happy",
            ],
            SET0: [
                "remove the smile",
                "make the face serious",
                "make the person not smile",
                "make the person look serious",
                "make the face not smiling",
            ],
        },
        "Eyeglasses": {
            SET1: [
                "add eyeglasses",
                "add glasses",
                "make the person wear glasses",
                "make the person wear eyeglasses",
                "put glasses on the person",
            ],
            SET0: [
                "remove the glasses",
                "remove eyeglasses",
                "make the person not wear glasses",
                "make the person not wear eyeglasses",
                "take off the glasses",
            ],
        },

        # 后续扩展四属性时可以直接打开使用
        "Young": {
            SET1: [
                "make the person look young",
                "make the face younger",
                "make the person younger",
            ],
            SET0: [
                "make the person look old",
                "make the face older",
                "make the person older",
            ],
        },
        "Blond_Hair": {
            SET1: [
                "make the hair blond",
                "change the hair to blond",
                "add blond hair",
            ],
            SET0: [
                "remove blond hair",
                "make the hair not blond",
                "change the hair away from blond",
            ],
        },
    }

    return templates


def generate_instruction_samples(selected_attrs: List[str]) -> List[Dict]:
    """
    自动构造文本指令数据集。

    返回样本格式：
        {
            "text": str,
            "labels": List[int],  # 长度等于属性数 K
        }

    对于 MVP:
        selected_attrs = ["Smiling", "Eyeglasses"]

    label 约定：
        KEEP = 0
        SET0 = 1
        SET1 = 2
    """
    phrase_templates = get_attr_phrase_templates()

    for attr in selected_attrs:
        if attr not in phrase_templates:
            raise ValueError(f"No text templates defined for attribute: {attr}")

    samples = []
    num_attrs = len(selected_attrs)

    # 1. 不编辑指令：所有属性 KEEP
    no_change_texts = [
        "keep the image unchanged",
        "do not change anything",
        "keep everything the same",
        "make no changes",
    ]

    for text in no_change_texts:
        samples.append({
            "text": text,
            "labels": [KEEP for _ in range(num_attrs)],
        })

    # 2. 单属性编辑：被提到的属性 SET0/SET1，其余属性 KEEP
    for attr_idx, attr in enumerate(selected_attrs):
        for action in [SET0, SET1]:
            for phrase in phrase_templates[attr][action]:
                labels = [KEEP for _ in range(num_attrs)]
                labels[attr_idx] = action

                samples.append({
                    "text": phrase,
                    "labels": labels,
                })

    # 3. 双属性组合编辑：用于 MVP 的 Smiling + Eyeglasses
    # 对于 K>2，也生成任意两属性组合。
    for i in range(num_attrs):
        for j in range(i + 1, num_attrs):
            attr_i = selected_attrs[i]
            attr_j = selected_attrs[j]

            for action_i in [SET0, SET1]:
                for action_j in [SET0, SET1]:
                    phrases_i = phrase_templates[attr_i][action_i]
                    phrases_j = phrase_templates[attr_j][action_j]

                    # 不需要所有笛卡尔积都用，避免数据太重复；
                    # 但 MVP 很小，这里取全部组合，便于模型充分拟合模板。
                    for pi in phrases_i:
                        for pj in phrases_j:
                            labels = [KEEP for _ in range(num_attrs)]
                            labels[i] = action_i
                            labels[j] = action_j

                            samples.append({
                                "text": f"{pi} and {pj}",
                                "labels": labels,
                            })

                            # 加一个反向语序，增强鲁棒性
                            samples.append({
                                "text": f"{pj} and {pi}",
                                "labels": labels,
                            })

    # 去重
    unique = {}
    for s in samples:
        key = s["text"]
        unique[key] = s

    samples = list(unique.values())
    return samples


def build_vocab(samples: List[Dict], min_freq: int = 1) -> Dict[str, int]:
    """
    根据自动生成的文本样本构造词表。
    """
    counter = {}

    for sample in samples:
        for tok in tokenize(sample["text"]):
            counter[tok] = counter.get(tok, 0) + 1

    vocab = {
        SPECIAL_TOKENS["PAD"]: 0,
        SPECIAL_TOKENS["UNK"]: 1,
        SPECIAL_TOKENS["CLS"]: 2,
    }

    for tok, freq in sorted(counter.items()):
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)

    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将文本编码为 token ids 和 attention mask。

    返回:
        input_ids: [max_len]
        attention_mask: [max_len]
            1 表示真实 token
            0 表示 padding
    """
    tokens = [SPECIAL_TOKENS["CLS"]] + tokenize(text)
    ids = [vocab.get(tok, vocab[SPECIAL_TOKENS["UNK"]]) for tok in tokens]

    if len(ids) > max_len:
        ids = ids[:max_len]

    attention_mask = [1] * len(ids)

    pad_id = vocab[SPECIAL_TOKENS["PAD"]]
    while len(ids) < max_len:
        ids.append(pad_id)
        attention_mask.append(0)

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


class TextInstructionDataset(Dataset):
    """
    文本指令数据集。

    每个样本：
        input_ids: [L]
        attention_mask: [L]
        labels: [K]
    """

    def __init__(
        self,
        samples: List[Dict],
        vocab: Dict[str, int],
        max_len: int,
    ):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        input_ids, attention_mask = encode_text(
            sample["text"],
            vocab=self.vocab,
            max_len=self.max_len,
        )

        labels = torch.tensor(sample["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": sample["text"],
        }


def split_samples(samples: List[Dict], train_ratio: float = 0.8, seed: int = 42):
    """
    将自动生成的模板样本划分为 train / valid。
    """
    rng = random.Random(seed)
    samples = samples.copy()
    rng.shuffle(samples)

    n_train = int(len(samples) * train_ratio)

    train_samples = samples[:n_train]
    valid_samples = samples[n_train:]

    return train_samples, valid_samples