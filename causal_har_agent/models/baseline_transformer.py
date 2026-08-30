"""
Standard Unconstrained Transformer Architecture for HAR Time-Series Classification.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sensor sequence coordinates."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to input tensor of shape (B, T, d_model)."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class StandardMultiHeadAttention(nn.Module):
    """Standard Multi-Head Scaled Dot-Product Self-Attention."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing standard self-attention.

        Args:
            x: Tensor of shape (B, T, d_model).
            mask: Optional attention mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (output_tensor, attention_weights)
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.out_proj(context)

        return output, attn_weights


class BaselineTransformerBlock(nn.Module):
    """Transformer Encoder block containing Multi-Head Attention and Position-wise FeedForward."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = StandardMultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN Self Attention
        residual = x
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(normed_x)
        x = residual + self.dropout(attn_out)

        # Feedforward
        residual = x
        normed_x = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(normed_x))))
        x = residual + self.dropout(ff_out)

        return x, attn_weights


class BaselineHARTransformer(nn.Module):
    """
    Standard Transformer architecture for Human Activity Recognition.
    Serves as the unconstrained baseline benchmark.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model

        # 1D Convolution input projection
        self.input_projection = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            BaselineTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of baseline model.

        Args:
            x: Input tensor of shape (B, T, C).
            return_attention: If True, returns attention weights from last layer.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (logits, last_layer_attention)
        """
        # Linear embedding over sensor channels per timestep
        h = self.input_projection(x)
        h = self.pos_encoder(h)

        last_attn = None
        for layer in self.layers:
            h, last_attn = layer(h)

        # Global average pooling across temporal dimension
        pooled = torch.mean(h, dim=1)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, last_attn
        return logits, None
