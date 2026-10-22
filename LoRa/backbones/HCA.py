# -*- coding: utf-8 -*-
"""Hermitian Covariance Attention (HCA), ported from HermNet-AMC.

Identical to the WiSig copy; kept self-contained for the Lora pipeline.
Operates on complex tensor [B, C, L]: builds the C x C Hermitian
covariance across L, matrix square root via Newton-Schulz, then
diagonal + upper-triangular features -> per-channel attention.

Property: a global phase rotation e^{jphi} cancels exactly in z @ z^H.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HermitianCovarianceAttention(nn.Module):
    """Second-order channel attention using full Hermitian covariance."""

    def __init__(self, channels, reduction=8, dropout=0.0,
                 sqrt_iter=3, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.sqrt_iter = sqrt_iter
        self.eps = eps

        hidden = max(2, channels // reduction)
        self.diag_proj = nn.Linear(channels, hidden, bias=False)
        self.offdiag_proj = nn.Linear(
            channels * (channels - 1), hidden, bias=False
        )
        self.fuse = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden, channels, bias=False),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

        upper_i, upper_j = torch.triu_indices(channels, channels, offset=1)
        self.register_buffer("upper_i", upper_i, persistent=False)
        self.register_buffer("upper_j", upper_j, persistent=False)

    @staticmethod
    def _newton_schulz_sqrt(
            matrix: torch.Tensor, num_iter: int) -> torch.Tensor:
        batch_size, channels, _ = matrix.shape
        dtype, device = matrix.dtype, matrix.device
        trace = matrix.diagonal(
            dim1=-2, dim2=-1
        ).real.sum(dim=-1).clamp(min=1e-6)
        matrix_norm = matrix / trace.view(batch_size, 1, 1).to(dtype)

        identity = torch.eye(
            channels, dtype=dtype, device=device
        ).expand(batch_size, channels, channels)
        y = matrix_norm
        z = identity.clone()
        for _ in range(num_iter):
            update = 0.5 * (3.0 * identity - z @ y)
            y = y @ update
            z = update @ z
        return y * trace.sqrt().view(batch_size, 1, 1).to(dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = z.shape

        covariance = torch.einsum(
            "bcl,bdl->bcd", z, z.conj()
        ) / length
        trace = covariance.diagonal(
            dim1=-2, dim2=-1
        ).real.sum(dim=-1)
        regularizer = (
            self.eps * trace / channels
        ).view(batch_size, 1, 1).to(covariance.dtype)
        identity = torch.eye(
            channels, dtype=covariance.dtype, device=covariance.device
        ).expand(batch_size, channels, channels)
        covariance = covariance + regularizer * identity

        covariance_sqrt = self._newton_schulz_sqrt(
            covariance, self.sqrt_iter
        )

        diagonal = covariance_sqrt.diagonal(
            dim1=-2, dim2=-1
        ).real
        diagonal = F.normalize(diagonal, dim=-1, eps=self.eps)
        diagonal_feature = self.diag_proj(diagonal)

        upper = covariance_sqrt[:, self.upper_i, self.upper_j]
        offdiagonal = torch.cat([upper.real, upper.imag], dim=-1)
        offdiagonal = F.normalize(offdiagonal, dim=-1, eps=self.eps)
        offdiagonal_feature = self.offdiag_proj(offdiagonal)

        attention_feature = torch.cat(
            [diagonal_feature, offdiagonal_feature], dim=-1
        )
        weights = self.dropout(
            self.fuse(attention_feature)
        ).unsqueeze(-1)
        return z * weights
