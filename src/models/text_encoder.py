import torch
import torch.nn as nn


class TinyTransformerTextEncoder(nn.Module):
    """
    Tiny Transformer Encoder for text instruction understanding.

    输入:
        input_ids: [B, L]
        attention_mask: [B, L]
            1 表示真实 token
            0 表示 padding

    输出:
        logits: [B, K, 3]
            K 是属性数量；
            3 对应 KEEP / SET0 / SET1。
    """

    def __init__(
        self,
        vocab_size: int,
        num_attrs: int,
        max_len: int = 32,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_attrs = num_attrs
        self.max_len = max_len
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

        # 每个属性一个三分类头
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, 3)
            for _ in range(num_attrs)
        ])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids:
            [B, L]

        attention_mask:
            [B, L], 1 for real tokens, 0 for padding

        return:
            logits: [B, K, 3]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)

        # PyTorch Transformer 的 src_key_padding_mask:
        # True 表示这个位置是 padding，需要被 mask 掉。
        src_key_padding_mask = attention_mask == 0

        encoded = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )

        encoded = self.layer_norm(encoded)

        # 使用 <cls> 位置表示整句语义
        cls_repr = encoded[:, 0, :]  # [B, D]

        logits_per_attr = []
        for head in self.heads:
            logits_per_attr.append(head(cls_repr))  # [B, 3]

        logits = torch.stack(logits_per_attr, dim=1)  # [B, K, 3]
        return logits


def build_text_encoder(
    vocab_size: int,
    num_attrs: int,
    max_len: int = 32,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    ff_dim: int = 256,
    dropout: float = 0.1,
):
    return TinyTransformerTextEncoder(
        vocab_size=vocab_size,
        num_attrs=num_attrs,
        max_len=max_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    )