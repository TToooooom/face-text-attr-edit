import torch
import torch.nn.functional as F


REGION_BY_ATTR = {
    "Smiling": "mouth",
    "Eyeglasses": "eyes",
    "Young": "face",
    "Blond_Hair": "hair",
    "Black_Hair": "hair",
    "Brown_Hair": "hair",
    "Bangs": "hair",
    "Male": "face",
}


def _rect_mask(batch_size, h, w, y1, y2, x1, x2, device):
    mask = torch.zeros(batch_size, 1, h, w, device=device)
    yy1, yy2 = int(y1 * h), int(y2 * h)
    xx1, xx2 = int(x1 * w), int(x2 * w)
    mask[:, :, yy1:yy2, xx1:xx2] = 1.0
    return mask


def _soften_mask(mask, kernel_size=15, repeat=2):
    """
    用平均池化近似软化边界，避免硬边界融合痕迹。
    """
    pad = kernel_size // 2
    out = mask
    for _ in range(repeat):
        out = F.avg_pool2d(out, kernel_size=kernel_size, stride=1, padding=pad)
    return out.clamp(0.0, 1.0)


def build_region_mask(region, batch_size, h, w, device):
    """
    基于 CelebA 对齐人脸的经验区域。
    输入图像经过 CenterCrop(178) + Resize 后，人脸位置相对稳定。
    """
    if region == "eyes":
        mask = _rect_mask(batch_size, h, w, 0.25, 0.48, 0.12, 0.88, device)

    elif region == "mouth":
        mask = _rect_mask(batch_size, h, w, 0.52, 0.82, 0.22, 0.78, device)

    elif region == "hair":
        mask = _rect_mask(batch_size, h, w, 0.00, 0.48, 0.03, 0.97, device)

    elif region == "face":
        mask = _rect_mask(batch_size, h, w, 0.10, 0.92, 0.10, 0.90, device)

    else:
        mask = torch.ones(batch_size, 1, h, w, device=device)

    return _soften_mask(mask)


def build_attribute_edit_mask(attr_names, edit_mask, image_size):
    """
    根据本 batch 中哪些属性被编辑，生成目标区域 mask。

    参数:
        attr_names: List[str], 长度 K
        edit_mask: [B, K], 0/1，1 表示该属性本次需要编辑
        image_size: int

    返回:
        region_mask: [B, 1, H, W]
    """
    device = edit_mask.device
    bsz, num_attrs = edit_mask.shape
    h = w = image_size

    total_mask = torch.zeros(bsz, 1, h, w, device=device)

    for j, attr in enumerate(attr_names):
        region = REGION_BY_ATTR.get(attr, "face")
        region_mask = build_region_mask(region, bsz, h, w, device)

        active = edit_mask[:, j].view(bsz, 1, 1, 1)
        total_mask = torch.maximum(total_mask, region_mask * active)

    return total_mask.clamp(0.0, 1.0)


def make_keep_mask(edit_mask):
    """
    edit_mask: [B, K]
    return keep_mask: [B, K]
    """
    return 1.0 - edit_mask