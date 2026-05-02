import os
import random
from PIL import Image, ImageDraw
import numpy as np


ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]


def make_simple_face_image(index: int, smiling: bool, eyeglasses: bool):
    """
    生成一张简单的“假人脸”图像，尺寸模拟 CelebA 对齐图像：178 x 218。
    这不是训练数据，只用于本地测试 Dataset 和 transform。
    """
    width, height = 178, 218

    bg_color = (
        220 + random.randint(-10, 10),
        220 + random.randint(-10, 10),
        220 + random.randint(-10, 10),
    )

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # 脸
    face_color = (235, 190, 160)
    draw.ellipse((39, 35, 139, 155), fill=face_color, outline=(80, 60, 50), width=2)

    # 眼睛
    draw.ellipse((65, 80, 73, 88), fill=(20, 20, 20))
    draw.ellipse((105, 80, 113, 88), fill=(20, 20, 20))

    # 眼镜
    if eyeglasses:
        draw.rectangle((58, 74, 80, 94), outline=(0, 0, 0), width=2)
        draw.rectangle((98, 74, 120, 94), outline=(0, 0, 0), width=2)
        draw.line((80, 84, 98, 84), fill=(0, 0, 0), width=2)

    # 嘴巴
    if smiling:
        draw.arc((72, 98, 108, 130), start=10, end=170, fill=(150, 30, 30), width=3)
    else:
        draw.line((75, 118, 105, 118), fill=(150, 30, 30), width=3)

    # 写一个编号，方便确认不是同一张
    draw.text((5, 5), str(index), fill=(0, 0, 0))

    return img


def main():
    random.seed(42)
    np.random.seed(42)

    root = "data/celeba"
    img_dir = os.path.join(root, "img_align_celeba")

    os.makedirs(img_dir, exist_ok=True)

    num_images = 40

    smiling_idx = ATTR_NAMES.index("Smiling")
    eyeglasses_idx = ATTR_NAMES.index("Eyeglasses")

    # 生成图片和属性
    all_attrs = {}

    for i in range(1, num_images + 1):
        filename = f"{i:06d}.jpg"

        # 让 Smiling / Eyeglasses 四种组合都出现
        smiling = (i % 2 == 0)
        eyeglasses = ((i // 2) % 2 == 0)

        img = make_simple_face_image(i, smiling=smiling, eyeglasses=eyeglasses)
        img.save(os.path.join(img_dir, filename))

        # CelebA 原始属性格式是 -1 / 1
        attrs = [-1 for _ in ATTR_NAMES]
        attrs[smiling_idx] = 1 if smiling else -1
        attrs[eyeglasses_idx] = 1 if eyeglasses else -1

        # 其他属性随机给一点变化，但不重要
        for j in range(len(attrs)):
            if j not in [smiling_idx, eyeglasses_idx]:
                attrs[j] = random.choice([-1, 1])

        all_attrs[filename] = attrs

    # 写 list_attr_celeba.txt
    attr_path = os.path.join(root, "list_attr_celeba.txt")
    with open(attr_path, "w", encoding="utf-8") as f:
        f.write(f"{num_images}\n")
        f.write(" ".join(ATTR_NAMES) + "\n")

        for i in range(1, num_images + 1):
            filename = f"{i:06d}.jpg"
            attr_str = " ".join(str(v) for v in all_attrs[filename])
            f.write(f"{filename} {attr_str}\n")

    # 写 list_eval_partition.txt
    # 0: train, 1: valid, 2: test
    partition_path = os.path.join(root, "list_eval_partition.txt")
    with open(partition_path, "w", encoding="utf-8") as f:
        for i in range(1, num_images + 1):
            filename = f"{i:06d}.jpg"

            if i <= 28:
                split = 0
            elif i <= 34:
                split = 1
            else:
                split = 2

            f.write(f"{filename} {split}\n")

    print("=" * 80)
    print("Mock CelebA created successfully.")
    print(f"Root: {root}")
    print(f"Images: {img_dir}")
    print(f"Number of images: {num_images}")
    print(f"Attribute file: {attr_path}")
    print(f"Partition file: {partition_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()