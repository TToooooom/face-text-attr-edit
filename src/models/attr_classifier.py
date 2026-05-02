import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18AttrClassifier(nn.Module):
    """
    ResNet18 多标签人脸属性分类器。

    输入:
        x: [B, 3, H, W]

    输出:
        logits: [B, K]

    注意:
        这里输出的是 logits，不经过 sigmoid。
        训练时使用 BCEWithLogitsLoss。
        推理或计算准确率时再手动 sigmoid。
    """

    def __init__(
        self,
        num_attrs: int,
        pretrained: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features

        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_attrs),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_attrs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return logits


def build_attr_classifier(
    num_attrs: int,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> ResNet18AttrClassifier:
    model = ResNet18AttrClassifier(
        num_attrs=num_attrs,
        pretrained=pretrained,
        dropout=dropout,
    )
    return model