# -*- coding: utf-8 -*-
"""ResCISNet + HCA: ASKModule with a Hermitian Covariance Attention
module inserted after the Koopman spectral evolution.

Same design as the WiSig PatchNetHCA port: the evolved complex
spectrum [B, 2*max_k-1] is zero-padded to a multiple of `n_bands`,
split into contiguous sub-bands [B, n_bands, band_len], attended by
HCA (sub-band attention driven by Hermitian second-order statistics,
invariant to global phase rotation), then reshaped back before the
IFFT reconstruction.

CPU note: we use candidates up to 1024 (max_modes 2047) instead of
the default 4096 (max_modes 8191), because KoopmanLayer performs a
QR decomposition of an n_atoms x n_atoms complex matrix every forward
pass — 8191^2 is GPU-only territory. The author's released run
converged to k=512, so the reduced candidate set still covers the
empirically relevant range.
"""

import torch
import torch.nn as nn

from backbones.ResCISNet import ASKModule, ResCISNet
from backbones.HCA import HermitianCovarianceAttention


class ASKModuleHCA(ASKModule):

    def __init__(self, *args, n_bands=32, hca_reduction=2, hca_dropout=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bands = n_bands
        n_modes = self.max_modes
        self.pad = (n_bands - n_modes % n_bands) % n_bands
        self.padded_modes = n_modes + self.pad
        self.band_len = self.padded_modes // n_bands
        self.hca = HermitianCovarianceAttention(
            n_bands, reduction=hca_reduction, dropout=hca_dropout
        )

    def forward(self, x, tau=1.0):
        B, C, L = x.shape

        x_complex = torch.complex(x[:, 0], x[:, 1])
        Xf = torch.fft.fft(x_complex, dim=-1)
        Xf_shift = torch.fft.fftshift(Xf, dim=-1)

        Xf_masked, selected_k = self.cutoff(Xf_shift, tau=tau)

        c = self.center
        Xf_low = Xf_masked[:, c - self.max_k + 1 : c + self.max_k]

        Xf_evolved = self.koopman(Xf_low)

        # ---- HCA over spectral sub-bands ----
        if self.pad:
            Xf_evolved = torch.cat(
                [Xf_evolved,
                 torch.zeros(B, self.pad, dtype=Xf_evolved.dtype,
                             device=Xf_evolved.device)],
                dim=1)
        bands = Xf_evolved.view(B, self.n_bands, self.band_len)
        bands = self.hca(bands)
        Xf_att = bands.reshape(B, self.padded_modes)
        if self.pad:
            Xf_att = Xf_att[:, :self.max_modes]
        # -------------------------------------

        Xf_full_shift = torch.zeros(B, L, dtype=torch.cfloat, device=x.device)
        Xf_full_shift[:, c - self.max_k + 1 : c + self.max_k] = Xf_att

        Xf_full = torch.fft.ifftshift(Xf_full_shift, dim=-1)
        x_recon = torch.fft.ifft(Xf_full, dim=-1)
        energy = torch.sum(x_recon.abs().pow(2), dim=-1, keepdim=True)
        x_normalized = x_recon / (torch.sqrt(energy) + 1e-8)

        return torch.stack([x_normalized.real, x_normalized.imag], dim=1)


class ResCISNetHCA(ResCISNet):

    def __init__(self, num_classes, win_len=128, crop_ratio=0.3,
                 seq_len=8192, candidates=None, n_bands=32,
                 hca_reduction=2):
        super().__init__(num_classes=num_classes, win_len=win_len,
                         crop_ratio=crop_ratio, seq_len=seq_len,
                         candidates=candidates)
        # replace the plain ASK module with the HCA-augmented one
        self.ask = ASKModuleHCA(seq_len=seq_len, candidates=candidates,
                                n_bands=n_bands, hca_reduction=hca_reduction)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = ResCISNetHCA(num_classes=10, seq_len=8192,
                     candidates=[128, 256, 512, 1024])
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 2, 8192))
    print("logits:", tuple(out.shape))
    print("params: {:,}".format(sum(p.numel() for p in m.parameters())))
