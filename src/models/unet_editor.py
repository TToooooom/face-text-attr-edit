import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, norm=True):
        super().__init__()

        if down:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)

        layers = [conv]

        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))

        layers.append(nn.ReLU(inplace=True) if not down else nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MaskGuidedUNetEditor(nn.Module):
    """
    Mask-Guided Conditional U-Net Residual Editor.

    输入:
        x:         [B, 3, H, W]
        c_trg:     [B, K]
        edit_mask: [B, K], 1 表示该属性需要编辑
        region_mask: [B, 1, H, W]

    输出:
        x_out: [B, 3, H, W]
        alpha: [B, 1, H, W]
        raw: [B, 3, H, W]
    """

    def __init__(
        self,
        c_dim: int,
        conv_dim: int = 64,
        delta_scale: float = 1.0,
    ):
        super().__init__()

        self.c_dim = c_dim
        self.delta_scale = delta_scale

        in_ch = 3 + c_dim + c_dim + 1

        self.down1 = ConvBlock(in_ch, conv_dim, down=True, norm=False)       # 128 -> 64
        self.down2 = ConvBlock(conv_dim, conv_dim * 2, down=True)            # 64 -> 32
        self.down3 = ConvBlock(conv_dim * 2, conv_dim * 4, down=True)        # 32 -> 16
        self.down4 = ConvBlock(conv_dim * 4, conv_dim * 8, down=True)        # 16 -> 8

        self.bottleneck = nn.Sequential(
            nn.Conv2d(conv_dim * 8, conv_dim * 8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(conv_dim * 8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim * 8, conv_dim * 8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(conv_dim * 8, affine=True),
            nn.ReLU(inplace=True),
        )

        self.up4 = ConvBlock(conv_dim * 8, conv_dim * 4, down=False)
        self.up3 = ConvBlock(conv_dim * 8, conv_dim * 2, down=False)
        self.up2 = ConvBlock(conv_dim * 4, conv_dim, down=False)
        self.up1 = ConvBlock(conv_dim * 2, conv_dim, down=False)

        self.out_conv = nn.Conv2d(conv_dim, 4, kernel_size=7, stride=1, padding=3)

    def forward(self, x, c_trg, edit_mask, region_mask):
        b, _, h, w = x.shape

        c_map = c_trg.view(b, self.c_dim, 1, 1).expand(b, self.c_dim, h, w)
        e_map = edit_mask.view(b, self.c_dim, 1, 1).expand(b, self.c_dim, h, w)

        inp = torch.cat([x, c_map, e_map, region_mask], dim=1)

        d1 = self.down1(inp)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        z = self.bottleneck(d4)

        u4 = self.up4(z)
        u3 = self.up3(torch.cat([u4, d3], dim=1))
        u2 = self.up2(torch.cat([u3, d2], dim=1))
        u1 = self.up1(torch.cat([u2, d1], dim=1))

        out = self.out_conv(u1)

        raw = torch.tanh(out[:, :3])
        alpha = torch.sigmoid(out[:, 3:4])

        # 强制只在目标区域编辑
        alpha = alpha * region_mask

        x_out = alpha * raw + (1.0 - alpha) * x
        x_out = x_out.clamp(-1.0, 1.0)

        return x_out, alpha, raw


def build_unet_editor(c_dim, g_conv_dim=64, delta_scale=1.0):
    return MaskGuidedUNetEditor(
        c_dim=c_dim,
        conv_dim=g_conv_dim,
        delta_scale=delta_scale,
    )