import re
from typing import Dict, List


KEEP = 0
SET0 = 1
SET1 = 2


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_instruction_to_actions(text: str, attr_names: List[str]) -> Dict[str, int]:
    """
    将自然语言解析为每个属性的动作:
        KEEP = 0
        SET0 = 1
        SET1 = 2

    支持中英文关键词。
    """
    text = normalize(text)

    actions = {attr: KEEP for attr in attr_names}

    # Smiling
    if "smile" in text or "happy" in text or "微笑" in text or "笑" in text:
        if "remove" in text or "not smile" in text or "serious" in text or "不要笑" in text or "无表情" in text:
            actions["Smiling"] = SET0
        else:
            actions["Smiling"] = SET1

    if "serious" in text or "neutral expression" in text or "无表情" in text:
        if "Smiling" in actions:
            actions["Smiling"] = SET0

    # Eyeglasses
    if "glasses" in text or "eyeglasses" in text or "眼镜" in text:
        if "remove" in text or "take off" in text or "not wear" in text or "去掉" in text or "摘掉" in text or "不戴" in text:
            actions["Eyeglasses"] = SET0
        else:
            actions["Eyeglasses"] = SET1

    # Young
    if "young" in text or "younger" in text or "年轻" in text:
        actions["Young"] = SET1

    if "old" in text or "older" in text or "年老" in text or "变老" in text:
        if "Young" in actions:
            actions["Young"] = SET0

    # Hair colors
    hair_color_attrs = ["Blond_Hair", "Black_Hair", "Brown_Hair"]

    if "blond" in text or "blonde" in text or "金发" in text:
        for a in hair_color_attrs:
            if a in actions:
                actions[a] = SET0
        if "Blond_Hair" in actions:
            actions["Blond_Hair"] = SET1

    if "black hair" in text or "黑发" in text:
        for a in hair_color_attrs:
            if a in actions:
                actions[a] = SET0
        if "Black_Hair" in actions:
            actions["Black_Hair"] = SET1

    if "brown hair" in text or "棕发" in text or "褐色头发" in text:
        for a in hair_color_attrs:
            if a in actions:
                actions[a] = SET0
        if "Brown_Hair" in actions:
            actions["Brown_Hair"] = SET1

    # Bangs
    if "bangs" in text or "刘海" in text:
        if "remove" in text or "去掉" in text or "不要" in text:
            actions["Bangs"] = SET0
        else:
            actions["Bangs"] = SET1

    # Male
    if "male" in text or "man" in text or "男性" in text or "男人" in text:
        if "female" not in text and "woman" not in text:
            actions["Male"] = SET1

    if "female" in text or "woman" in text or "女性" in text or "女人" in text:
        if "Male" in actions:
            actions["Male"] = SET0

    return actions


def actions_to_tensor(actions: Dict[str, int], attr_names: List[str]):
    import torch
    return torch.tensor([actions[attr] for attr in attr_names], dtype=torch.long)