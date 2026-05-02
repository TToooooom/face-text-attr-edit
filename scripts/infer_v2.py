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
from src.data.attribute_masks import build_attribute_edit_mask
from src.models.attr_classifier import build_attr_classifier
from src.models.unet_editor import build_unet_editor
from src.text.rule_parser import parse_instruction_to_actions, SET0, SET1


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def denormalize(x):
    return (x + 1.0) / 2.0


def load_image(path, image_size, crop_size, device):
    image = Image.open(path).convert("RGB")
    transform = build_celeba_transform(
        image_size=image_size,
        crop_size=crop_size,
        mode="eval",
    )
    return transform(image).unsqueeze(0).to(device)


def load_attr_classifier(path, num_attrs, device):
    ckpt = safe_torch_load(path, map_location=device)

    model = build_attr_classifier(
        num_attrs=num_attrs,
        pretrained=False,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

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


@torch.no_grad()
def run_inference(args):
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    attr_names = data_cfg["selected_attrs"]
    image_size = data_cfg["image_size"]
    crop_size = data_cfg["crop_size"]
    num_attrs = len(attr_names)

    os.makedirs(args.output_dir, exist_ok=True)

    attr_model = load_attr_classifier(args.attr_ckpt, num_attrs, device)
    G, ckpt_cfg, ckpt_attrs = load_editor(args.editor_ckpt, device)

    x = load_image(args.image, image_size, crop_size, device)

    logits = attr_model(x)
    probs = torch.sigmoid(logits)
    c_org = (probs >= args.attr_threshold).float()

    actions = parse_instruction_to_actions(args.text, attr_names)

    c_trg = c_org.clone()
    edit_mask = torch.zeros_like(c_org)

    for j, attr in enumerate(attr_names):
        action = actions[attr]
        if action == SET0:
            c_trg[:, j] = 0.0
            edit_mask[:, j] = 1.0
        elif action == SET1:
            c_trg[:, j] = 1.0
            edit_mask[:, j] = 1.0

    # 发色互斥
    hair_attrs = ["Blond_Hair", "Black_Hair", "Brown_Hair"]
    set_hair = [a for a in hair_attrs if a in attr_names and actions.get(a, 0) == SET1]
    if len(set_hair) > 0:
        chosen = set_hair[0]
        for a in hair_attrs:
            if a in attr_names:
                idx = attr_names.index(a)
                c_trg[:, idx] = 0.0
                edit_mask[:, idx] = 1.0
        c_trg[:, attr_names.index(chosen)] = 1.0

    region_mask = build_attribute_edit_mask(attr_names, edit_mask, image_size)

    x_fake, alpha, raw = G(x, c_trg, edit_mask, region_mask)

    stem = Path(args.image).stem

    save_image(denormalize(x.cpu()), os.path.join(args.output_dir, f"{stem}_original.jpg"))
    save_image(denormalize(x_fake.cpu()), os.path.join(args.output_dir, f"{stem}_edited.jpg"))
    save_image(alpha.cpu(), os.path.join(args.output_dir, f"{stem}_alpha.jpg"))
    save_image(
        denormalize(torch.cat([x.cpu(), x_fake.cpu()], dim=0)),
        os.path.join(args.output_dir, f"{stem}_grid.jpg"),
        nrow=2,
    )

    print("=" * 80)
    print("V2 Inference")
    print("=" * 80)
    print(f"Text: {args.text}")
    print("Source attributes:")
    for j, attr in enumerate(attr_names):
        print(f"  {attr:12s}: value={int(c_org[0, j].item())}, prob={probs[0, j].item():.4f}")

    print("Actions:")
    for attr in attr_names:
        print(f"  {attr:12s}: {actions[attr]}")

    print("Target attributes:")
    for j, attr in enumerate(attr_names):
        print(f"  {attr:12s}: {int(c_trg[0, j].item())}")

    print(f"Saved to: {args.output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/v2_server_128_8attr.yaml")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)

    parser.add_argument("--attr_ckpt", type=str, default="checkpoints/attr_classifier_v2/best.pt")
    parser.add_argument("--editor_ckpt", type=str, default="checkpoints/unet_editor_v2/final.pt")

    parser.add_argument("--output_dir", type=str, default="outputs/inference_v2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--attr_threshold", type=float, default=0.5)

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()