"""
Causal-Guided Transformer Architecture for Robust Human Activity Recognition.
Integrates LLM-Discovered Structural Causal Graphs into Attention Mechanisms.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_har_agent.causal.graph_structure import CausalGraph
from causal_har_agent.models.baseline_transformer import PositionalEncoding


class CausalAttention(nn.Module):
    """
    Multi-Head Attention mechanism regularized by a Causal Adjacency Prior.
    Applies the mathematical formulation:
        A_causal = Softmax( (Q K^T / sqrt(d_k)) + lambda * M_causal )
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
        causal_weight_init: float = 1.0,
        learnable_weight: bool = True,
    ) -> None:
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

        if learnable_weight:
            self.causal_lambda = nn.Parameter(torch.tensor(causal_weight_init))
        else:
            self.register_buffer("causal_lambda", torch.tensor(causal_weight_init))

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal bias injection.

        Args:
            x: Input tensor of shape (B, T, d_model).
            causal_mask: Tensor of shape (T, T) or broadcastable (1, 1, T, T) representing
                         causal flow prior.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (output_tensor, attention_weights)
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Inject causal bias prior
        if causal_mask is not None:
            if causal_mask.ndim == 2:
                # Shape: (1, 1, T, T)
                bias = causal_mask.unsqueeze(0).unsqueeze(0)
            elif causal_mask.ndim == 3:
                bias = causal_mask.unsqueeze(1)
            else:
                bias = causal_mask

            scores = scores + self.causal_lambda * bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.out_proj(context)

        return output, attn_weights


class CausalTransformerBlock(nn.Module):
    """Transformer Encoder block integrating Causal-Guided Attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        causal_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = CausalAttention(
            d_model, nhead, dropout, causal_weight_init=causal_weight_init
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(normed_x, causal_mask)
        x = residual + self.dropout(attn_out)

        residual = x
        normed_x = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(normed_x))))
        x = residual + self.dropout(ff_out)

        return x, attn_weights


class CausalGuidedHARTransformer(nn.Module):
    """
    Causal-HAR-Agent: Deep Neural Architecture with Structural Causal Model Induction.
    Maps physical sensor channels and temporal kinematics according to discovered DAGs.
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
        causal_graph: Optional[CausalGraph] = None,
        ordered_channels: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model

        # Channel-wise causal embedding layer
        self.channel_embedding = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            CausalTransformerBlock(
                d_model, nhead, dim_feedforward, dropout, causal_weight_init=1.2
            )
            for _ in range(num_layers)
        ])

        # Classification and Invariant Feature head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.ordered_channels = ordered_channels or [
            "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
            "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
            "waist_acc_x", "waist_acc_y", "waist_acc_z",
            "waist_gyro_x", "waist_gyro_y", "waist_gyro_z",
        ]

        self.causal_graph = causal_graph
        self.register_buffer("causal_adj_matrix", torch.zeros(in_channels, in_channels))

        if causal_graph is not None:
            self.set_causal_graph(causal_graph)

    def set_causal_graph(self, causal_graph: CausalGraph) -> None:
        """Updates the registered causal adjacency prior matrix."""
        self.causal_graph = causal_graph
        adj_np = causal_graph.get_causal_adjacency_matrix(
            self.ordered_channels[: self.in_channels]
        )
        self.causal_adj_matrix = torch.from_numpy(adj_np).float()

    def generate_temporal_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Constructs a temporal kinematic causal mask enforcing that forward causes precede
        backward effects (temporal Arrow of Time + kinematic forward delay).
        """
        # Lower-triangular causal autoregressive / forward kinematic mask
        # M_temp[i, j] = 0.0 if j <= i else -1e4
        temporal_mask = torch.triu(torch.full((seq_len, seq_len), -1e4, device=device), diagonal=1)

        # Allow immediate small backward window (anticipatory muscle pre-activation: 2 timesteps)
        for i in range(seq_len):
            for j in range(max(0, i - 15), min(seq_len, i + 3)):
                temporal_mask[i, j] = 0.0

        return temporal_mask

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Causal-Guided HAR model.

        Args:
            x: Sensor tensor of shape (B, T, C).
            return_attention: If True, returns causal attention weights.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (class_logits, attention_weights)
        """
        B, T, C = x.shape
        device = x.device

        # Channel-level causal projection
        # If causal adjacency matrix is set, weight channel inputs by causal parents
        if self.causal_adj_matrix.shape[0] == C:
            # Weighted channel propagation
            channel_weighted = torch.matmul(x, self.causal_adj_matrix.to(device).T)
            x_in = 0.7 * x + 0.3 * channel_weighted
        else:
            x_in = x

        h = self.channel_embedding(x_in)
        h = self.pos_encoder(h)

        causal_mask = self.generate_temporal_causal_mask(T, device)

        last_attn = None
        for layer in self.layers:
            h, last_attn = layer(h, causal_mask=causal_mask)

        # Invariant representation pooling
        pooled = torch.mean(h, dim=1)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, last_attn
        return logits, None
