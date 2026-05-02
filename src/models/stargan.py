import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    StarGAN 生成器中的残差块。

    输入输出通道数保持不变：
        [B, C, H, W] -> [B, C, H, W]
    """

    def __init__(self, dim: int):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
        )

    def forward(self, x):
        return x + self.main(x)


class StarGANGenerator(nn.Module):
    """
    StarGAN v1 风格生成器。

    输入:
        x: [B, 3, H, W]
        c: [B, K]

    做法:
        将属性向量 c 扩展成 [B, K, H, W]，
        与图像在通道维拼接：
            [B, 3 + K, H, W]
    """

    def __init__(
        self,
        image_channels: int = 3,
        c_dim: int = 2,
        conv_dim: int = 64,
        repeat_num: int = 6,
    ):
        super().__init__()

        layers = []

        # 初始卷积
        layers += [
            nn.Conv2d(
                image_channels + c_dim,
                conv_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
        ]

        # 下采样：64 -> 32 -> 16
        curr_dim = conv_dim
        for _ in range(2):
            layers += [
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # 残差块
        for _ in range(repeat_num):
            layers += [ResidualBlock(curr_dim)]

        # 上采样：16 -> 32 -> 64
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(curr_dim // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim //= 2

        # 输出图像，范围 [-1, 1]
        layers += [
            nn.Conv2d(
                curr_dim,
                image_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.Tanh(),
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        x: [B, 3, H, W]
        c: [B, K]
        """
        b, _, h, w = x.shape

        c_map = c.view(b, c.size(1), 1, 1)
        c_map = c_map.expand(b, c.size(1), h, w)

        x_in = torch.cat([x, c_map], dim=1)
        out = self.main(x_in)

        return out


class StarGANDiscriminator(nn.Module):
    """
    StarGAN v1 风格判别器。

    输入:
        x: [B, 3, H, W]

    输出:
        out_src: [B, 1, h, w]
            PatchGAN 风格真假分数，不经过 sigmoid。

        out_cls: [B, K]
            属性分类 logits，不经过 sigmoid。
    """

    def __init__(
        self,
        image_size: int = 64,
        image_channels: int = 3,
        c_dim: int = 2,
        conv_dim: int = 64,
        repeat_num: int = 4,
    ):
        super().__init__()

        layers = []

        layers += [
            nn.Conv2d(
                image_channels,
                conv_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.01, inplace=True),
        ]

        curr_dim = conv_dim

        for _ in range(1, repeat_num):
            layers += [
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.LeakyReLU(0.01, inplace=True),
            ]
            curr_dim *= 2

        self.main = nn.Sequential(*layers)

        # PatchGAN 真假头
        self.conv_src = nn.Conv2d(
            curr_dim,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # 属性分类头：用全局平均池化 + 线性层，更稳、更简单
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_cls = nn.Linear(curr_dim, c_dim)

    def forward(self, x: torch.Tensor):
        h = self.main(x)

        out_src = self.conv_src(h)

        pooled = self.avgpool(h).view(h.size(0), -1)
        out_cls = self.fc_cls(pooled)

        return out_src, out_cls


def build_stargan_generator(
    c_dim: int,
    g_conv_dim: int = 64,
    g_repeat_num: int = 6,
):
    return StarGANGenerator(
        image_channels=3,
        c_dim=c_dim,
        conv_dim=g_conv_dim,
        repeat_num=g_repeat_num,
    )


def build_stargan_discriminator(
    image_size: int,
    c_dim: int,
    d_conv_dim: int = 64,
    d_repeat_num: int = 4,
):
    return StarGANDiscriminator(
        image_size=image_size,
        image_channels=3,
        c_dim=c_dim,
        conv_dim=d_conv_dim,
        repeat_num=d_repeat_num,
    )