"""
Vision-Language Transformer for Cluster Classification.

Architecture:
1. Video Encoder (3D ResNet-18) → temporal tokens {v_1, ..., v_T}
2. Text Encoder (BERT/SBERT) → action token a
3. Joint sequence: [CLS, a, v_1, ..., v_T]
4. Transformer self-attention (4 layers, 4 heads)
5. Linear classifier on CLS token → 50-way prediction

Tensor shapes (example with batch=2, T=16 frames):
- Video input:    (B, 3, T, 224, 224) = (2, 3, 16, 224, 224)
- Video tokens:   (B, T', 256)        = (2, 2, 256)    # T'=T/8=2 after full ResNet
- Action token:   (B, 1, 256)         = (2, 1, 256)
- Joint sequence: (B, 1+1+T', 256)    = (2, 4, 256)    # [CLS, action, v1, v2]
- Output logits:  (B, 50)             = (2, 50)

Freezing strategy:
- Video encoder: freeze stem, layer1, layer2; train layer3, layer4
- Text encoder: freeze embeddings + first 10 layers; train last 2 layers
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18
from transformers import AutoModel


class R3DEncoder(nn.Module):
    """3D ResNet-18 video encoder with frozen early layers.

    Uses full network (stem + layer1-4), freezes early layers, trains later layers.
    - Freeze: stem, layer1, layer2
    - Train: layer3, layer4
    - Output: 512 channels from layer4

    With grid_size > 1, outputs spatial grid tokens instead of global pooling.
    """

    def __init__(self, pretrained: bool = True, freeze_early_layers: bool = True, grid_size: int = 1):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        model = r3d_18(weights=weights)
        self.stem = model.stem
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.out_dim = 512  # layer4 output channels
        self.grid_size = grid_size  # s: output s×s spatial tokens per timestep

        # Freeze early layers (stem, layer1, layer2) - only fine-tune layer3, layer4
        if freeze_early_layers and pretrained:
            for param in self.stem.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            # layer3, layer4 remain trainable

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, T'*s*s, D) tokens.

        With grid_size=1: global pooling, output (B, T/8, 512)
        With grid_size=2: 2x2 spatial grid, output (B, T/8*4, 512)
        """
        x = self.stem(video)   # (B, 64, T, 112, 112)
        x = self.layer1(x)     # (B, 64, T, 112, 112)
        x = self.layer2(x)     # (B, 128, T/2, 56, 56)
        x = self.layer3(x)     # (B, 256, T/4, 28, 28)
        x = self.layer4(x)     # (B, 512, T/8, 14, 14)

        s = self.grid_size
        if s == 1:
            # Global average pooling (original behavior)
            x = x.mean(dim=[3, 4])  # (B, 512, T/8)
            return x.transpose(1, 2)  # (B, T/8, 512)
        else:
            # Adaptive pooling to s×s grid, then flatten to tokens
            T_out = x.shape[2]  # T/8
            x = F.adaptive_avg_pool3d(x, output_size=(T_out, s, s))  # (B, 512, T', s, s)
            x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T', s, s, 512)
            x = x.view(x.shape[0], T_out * s * s, x.shape[-1])  # (B, T'*s*s, 512)
            return x


class TextEncoder(nn.Module):
    """BERT encoder with frozen early layers."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_early_layers: bool = True,
        num_trainable_layers: int = 2,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Freeze early layers
        if freeze_early_layers:
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
            num_layers = len(self.model.encoder.layer)
            freeze_until = num_layers - num_trainable_layers
            for i, layer in enumerate(self.model.encoder.layer):
                if i < freeze_until:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Returns projected CLS token (B, D)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)"""
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class VLTransformer(nn.Module):
    """
    Vision-Language Transformer for 50-way cluster classification.

    Architecture:
    1. Video encoder (3D ResNet-18) → temporal tokens (with optional spatial grid)
    2. Text encoder (BERT) → action token
    3. Joint sequence: [CLS, action, video_1, ..., video_N]
    4. Transformer (4 layers, 4 heads)
    5. Linear classifier on CLS token

    Regularization:
    - Video token dropout: randomly zero out video tokens during training
    - Modality dropout: occasionally zero out action token to force video usage
    """

    def __init__(
        self,
        num_classes: int = 50,
        hidden_dim: int = 256,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        text_model_name: str = "bert-base-uncased",
        freeze_early_layers: bool = True,
        num_trainable_text_layers: int = 2,
        max_video_tokens: int = 32,
        ablation_mode: str = "both",
        grid_size: int = 1,
        video_token_drop: float = 0.1,
        modality_drop: float = 0.1,
        use_temporal_delta: bool = True,
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.max_video_tokens = max_video_tokens
        self.video_token_drop = video_token_drop
        self.modality_drop = modality_drop
        self.grid_size = grid_size
        self.use_temporal_delta = use_temporal_delta

        # Encoders
        self.video_encoder = R3DEncoder(
            pretrained=True,
            freeze_early_layers=freeze_early_layers,
            grid_size=grid_size,
        )
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_early_layers=freeze_early_layers,
            num_trainable_layers=num_trainable_text_layers,
        )

        # Video projection
        self.video_proj = nn.Sequential(
            nn.Linear(self.video_encoder.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Temporal delta scale (learnable)
        if use_temporal_delta:
            self.delta_scale = nn.Parameter(torch.tensor(0.1))

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        max_seq_len = 2 + max_video_tokens  # CLS + action + video tokens
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_attention_heads, dropout)
            for _ in range(num_transformer_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        video: torch.Tensor,
        text_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W)
            text_inputs: dict with input_ids, attention_mask, token_type_ids
        Returns:
            (B, num_classes) logits
        """
        B = video.shape[0]

        # Encode video -> (B, N, D) where N = T'*s*s
        video_tokens = self.video_proj(self.video_encoder(video))

        # Inject temporal delta: add first-order difference to emphasize motion
        if self.use_temporal_delta and self.grid_size > 0:
            s2 = self.grid_size * self.grid_size
            N = video_tokens.shape[1]
            if N % s2 == 0 and N > s2:
                T_out = N // s2
                # Reshape to (B, T, P, D) where P = s*s patches per timestep
                v = video_tokens.view(B, T_out, s2, -1)
                # Compute temporal difference
                dv = torch.zeros_like(v)
                dv[:, 1:] = v[:, 1:] - v[:, :-1]
                # Add scaled delta back
                v = v + self.delta_scale * dv
                video_tokens = v.view(B, N, -1)

        # Encode text -> (B, D) -> (B, 1, D)
        action_token = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
            text_inputs["token_type_ids"],
        ).unsqueeze(1)

        # Ablation modes
        if self.ablation_mode == "video_only":
            action_token = torch.zeros_like(action_token)
        elif self.ablation_mode == "text_only":
            video_tokens = torch.zeros_like(video_tokens)

        # Training regularization: token dropout and modality dropout
        if self.training:
            # Video token dropout: randomly zero out some video tokens
            if self.video_token_drop > 0:
                keep_mask = torch.rand(video_tokens.shape[:2], device=video_tokens.device) > self.video_token_drop
                video_tokens = video_tokens * keep_mask.unsqueeze(-1)

            # Modality dropout: occasionally zero out action token to force video usage
            if self.modality_drop > 0 and torch.rand(1).item() < self.modality_drop:
                action_token = torch.zeros_like(action_token)

        # Build sequence: [CLS, action, video_1, ..., video_N]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, action_token, video_tokens], dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.final_norm(x)

        # Classify from CLS token
        return self.classifier(x[:, 0, :])


def build_model(
    num_classes: int = 50,
    hidden_dim: int = 256,
    num_transformer_layers: int = 4,
    num_attention_heads: int = 4,
    dropout: float = 0.1,
    text_model_name: str = "bert-base-uncased",
    freeze_early_layers: bool = True,
    num_trainable_text_layers: int = 2,
    max_video_tokens: int = 32,
    ablation_mode: str = "both",
    grid_size: int = 1,
    video_token_drop: float = 0.1,
    modality_drop: float = 0.1,
    use_temporal_delta: bool = True,
    **kwargs,
) -> VLTransformer:
    """Build VLTransformer model.

    Args:
        grid_size: Spatial grid size for video tokens (1=global, 2=2x2, 3=3x3)
        video_token_drop: Probability of dropping each video token during training
        modality_drop: Probability of dropping action token entirely during training
        use_temporal_delta: Whether to inject temporal difference signal
    """
    return VLTransformer(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_transformer_layers=num_transformer_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        text_model_name=text_model_name,
        freeze_early_layers=freeze_early_layers,
        num_trainable_text_layers=num_trainable_text_layers,
        max_video_tokens=max_video_tokens,
        ablation_mode=ablation_mode,
        grid_size=grid_size,
        video_token_drop=video_token_drop,
        modality_drop=modality_drop,
        use_temporal_delta=use_temporal_delta,
    )


if __name__ == "__main__":
    print("Building VLTransformer with grid_size=2...")
    # Test with grid_size=2: 16 frames -> T'=2, 2x2 grid -> 2*4=8 video tokens
    model = build_model(grid_size=2, max_video_tokens=8)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable:,} / {total:,} trainable ({100*trainable/total:.1f}%)")

    video = torch.randn(2, 3, 16, 224, 224)
    text = {
        "input_ids": torch.randint(0, 100, (2, 24)),
        "attention_mask": torch.ones(2, 24, dtype=torch.long),
        "token_type_ids": torch.zeros(2, 24, dtype=torch.long),
    }

    # Test training mode (with dropout)
    model.train()
    out_train = model(video, text)
    print(f"Training: video {video.shape} -> {out_train.shape}")

    # Test eval mode (no dropout)
    model.eval()
    with torch.no_grad():
        out_eval = model(video, text)
    print(f"Eval: video {video.shape} -> {out_eval.shape}")
    print(f"Video tokens: T'={16//8}, grid=2x2 -> {16//8 * 4} tokens")
    print("SUCCESS!")
