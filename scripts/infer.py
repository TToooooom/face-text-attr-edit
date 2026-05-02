import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
from PIL import Image
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import build_celeba_transform
from src.models.attr_classifier import build_attr_classifier
from src.models.text_encoder import build_text_encoder
from src.models.stargan import build_stargan_generator
from src.text.templates import encode_text, ID_TO_ACTION, KEEP, SET0, SET1


def safe_torch_load(path: str, map_location):
    """
    兼容不同 PyTorch 版本的 torch.load。
    某些新版本 torch.load 默认 weights_only 行为变化，显式设为 False 更稳。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """
    [-1, 1] -> [0, 1]
    """
    return (x + 1.0) / 2.0


def load_image(image_path: str, image_size: int, crop_size: int, device):
    """
    读取并预处理单张图像。

    返回:
        image_tensor: [1, 3, H, W]
    """
    image = Image.open(image_path).convert("RGB")

    transform = build_celeba_transform(
        image_size=image_size,
        crop_size=crop_size,
        mode="eval",
    )

    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


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


def load_text_encoder(ckpt_path: str, device):
    ckpt = safe_torch_load(ckpt_path, map_location=device)

    cfg = ckpt["config"]
    vocab = ckpt["vocab"]
    attr_names = ckpt["attr_names"]

    text_cfg = cfg["text"]

    model = build_text_encoder(
        vocab_size=len(vocab),
        num_attrs=len(attr_names),
        max_len=text_cfg["max_len"],
        embed_dim=text_cfg["embed_dim"],
        num_heads=text_cfg["num_heads"],
        num_layers=text_cfg["num_layers"],
        ff_dim=text_cfg["ff_dim"],
        dropout=text_cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, vocab, attr_names, text_cfg


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


@torch.no_grad()
def predict_source_attrs(attr_model, image_tensor, attr_threshold: float):
    """
    用图像属性分类器预测原图属性。

    返回:
        c_s: [1, K], 0/1 float tensor
        probs: [1, K], 概率
    """
    logits = attr_model(image_tensor)
    probs = torch.sigmoid(logits)
    c_s = (probs >= attr_threshold).float()

    return c_s, probs


@torch.no_grad()
def predict_text_actions(text_model, vocab, text: str, max_len: int, device):
    """
    用文本编码器预测每个属性的动作。

    返回:
        actions: [1, K], long tensor
        logits: [1, K, 3]
    """
    input_ids, attention_mask = encode_text(
        text=text,
        vocab=vocab,
        max_len=max_len,
    )

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    logits = text_model(input_ids, attention_mask)
    actions = torch.argmax(logits, dim=-1)

    return actions, logits


def compose_target_attrs(c_s: torch.Tensor, actions: torch.Tensor):
    """
    根据原属性 c_s 和文本动作 actions 合成目标属性 c_t。

    c_s:
        [1, K], 0/1

    actions:
        [1, K], 每个元素属于:
            KEEP = 0
            SET0 = 1
            SET1 = 2

    规则:
        KEEP: c_t_i = c_s_i
        SET0: c_t_i = 0
        SET1: c_t_i = 1
    """
    c_t = c_s.clone()

    c_t[actions == SET0] = 0.0
    c_t[actions == SET1] = 1.0

    return c_t


def print_inference_info(attr_names, attr_probs, c_s, actions, c_t, text):
    print("=" * 80)
    print("Inference Information")
    print("=" * 80)
    print(f"Text instruction: {text}")
    print("-" * 80)

    print("Predicted source attributes:")
    for i, name in enumerate(attr_names):
        prob = attr_probs[0, i].item()
        value = int(c_s[0, i].item())
        print(f"  {name:12s}: value={value}, prob={prob:.4f}")

    print("-" * 80)
    print("Predicted text actions:")
    for i, name in enumerate(attr_names):
        action_id = int(actions[0, i].item())
        action_name = ID_TO_ACTION[action_id]
        print(f"  {name:12s}: {action_name}")

    print("-" * 80)
    print("Target attributes:")
    for i, name in enumerate(attr_names):
        value = int(c_t[0, i].item())
        print(f"  {name:12s}: {value}")

    print("=" * 80)


@torch.no_grad()
def run_inference(
    image_path: str,
    text: str,
    config_path: str,
    attr_ckpt: str,
    text_ckpt: str,
    stargan_ckpt: str,
    output_dir: str,
    device: str,
    attr_threshold: float,
    save_rec: bool,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    attr_names = data_cfg["selected_attrs"]
    num_attrs = len(attr_names)

    image_size = data_cfg["image_size"]
    crop_size = data_cfg["crop_size"]

    device = torch.device(device)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型
    attr_model, attr_ckpt_attrs = load_attr_classifier(
        ckpt_path=attr_ckpt,
        num_attrs=num_attrs,
        device=device,
    )

    text_model, vocab, text_ckpt_attrs, text_cfg = load_text_encoder(
        ckpt_path=text_ckpt,
        device=device,
    )

    G, stargan_ckpt_attrs = load_stargan_generator(
        ckpt_path=stargan_ckpt,
        device=device,
    )

    # 2. 简单检查属性顺序是否一致
    if attr_ckpt_attrs is not None and attr_ckpt_attrs != attr_names:
        print(f"[Warning] attr classifier attrs {attr_ckpt_attrs} != config attrs {attr_names}")

    if text_ckpt_attrs != attr_names:
        print(f"[Warning] text encoder attrs {text_ckpt_attrs} != config attrs {attr_names}")

    if stargan_ckpt_attrs != attr_names:
        print(f"[Warning] stargan attrs {stargan_ckpt_attrs} != config attrs {attr_names}")

    # 3. 读取图像
    x_s = load_image(
        image_path=image_path,
        image_size=image_size,
        crop_size=crop_size,
        device=device,
    )

    # 4. 预测原图属性
    c_s, attr_probs = predict_source_attrs(
        attr_model=attr_model,
        image_tensor=x_s,
        attr_threshold=attr_threshold,
    )

    # 5. 预测文本动作
    actions, text_logits = predict_text_actions(
        text_model=text_model,
        vocab=vocab,
        text=text,
        max_len=text_cfg["max_len"],
        device=device,
    )

    # 6. 合成目标属性
    c_t = compose_target_attrs(
        c_s=c_s,
        actions=actions,
    )

    print_inference_info(
        attr_names=attr_names,
        attr_probs=attr_probs.detach().cpu(),
        c_s=c_s.detach().cpu(),
        actions=actions.detach().cpu(),
        c_t=c_t.detach().cpu(),
        text=text,
    )

    # 7. 生成编辑图像
    x_hat = G(x_s, c_t)

    # 8. 可选：重建图像
    if save_rec:
        x_rec = G(x_hat, c_s)
    else:
        x_rec = None

    # 9. 保存结果
    input_name = Path(image_path).stem

    edited_path = os.path.join(output_dir, f"{input_name}_edited.jpg")
    save_image(denormalize(x_hat.cpu()), edited_path)

    original_path = os.path.join(output_dir, f"{input_name}_original.jpg")
    save_image(denormalize(x_s.cpu()), original_path)

    if x_rec is not None:
        rec_path = os.path.join(output_dir, f"{input_name}_reconstructed.jpg")
        save_image(denormalize(x_rec.cpu()), rec_path)
    else:
        rec_path = None

    # 保存对比图：原图 | 编辑图 | 重建图
    if x_rec is not None:
        grid = torch.cat([x_s.cpu(), x_hat.cpu(), x_rec.cpu()], dim=0)
        grid_path = os.path.join(output_dir, f"{input_name}_grid.jpg")
        save_image(denormalize(grid), grid_path, nrow=3)
    else:
        grid = torch.cat([x_s.cpu(), x_hat.cpu()], dim=0)
        grid_path = os.path.join(output_dir, f"{input_name}_grid.jpg")
        save_image(denormalize(grid), grid_path, nrow=2)

    print("Saved outputs:")
    print(f"  Original: {original_path}")
    print(f"  Edited:   {edited_path}")
    if rec_path is not None:
        print(f"  Rec:      {rec_path}")
    print(f"  Grid:     {grid_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/mvp_64.yaml")

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)

    parser.add_argument("--attr_ckpt", type=str, default="checkpoints/attr_classifier/best.pt")
    parser.add_argument("--text_ckpt", type=str, default="checkpoints/text_encoder/best.pt")
    parser.add_argument("--stargan_ckpt", type=str, default="checkpoints/stargan/final.pt")

    parser.add_argument("--output_dir", type=str, default="outputs/inference")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--attr_threshold", type=float, default=0.5)

    parser.add_argument("--save_rec", action="store_true")

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    run_inference(
        image_path=args.image,
        text=args.text,
        config_path=args.config,
        attr_ckpt=args.attr_ckpt,
        text_ckpt=args.text_ckpt,
        stargan_ckpt=args.stargan_ckpt,
        output_dir=args.output_dir,
        device=device,
        attr_threshold=args.attr_threshold,
        save_rec=args.save_rec,
    )


if __name__ == "__main__":
    main()