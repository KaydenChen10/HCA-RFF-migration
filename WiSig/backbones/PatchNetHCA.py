"""PatchNet + HCA: ASKNet's PatchNet with a Hermitian Covariance
Attention module inserted in the spectral domain.

Placement rationale: HCA acts on the Koopman-evolved spectrum
[B, 2*max_k-1] (complex). The spectrum is split into `n_bands`
contiguous sub-bands -> [B, n_bands, band_len] complex, HCA attends
over sub-bands using their Hermitian second-order statistics, and the
attended spectrum is reshaped back before the IFFT reconstruction.

Because z @ z^H exactly cancels any global phase rotation, the
attention is driven by cross-band correlation structure (a stable
device fingerprint cue) instead of raw phase (dominated by
receiver/channel effects) — the property HermNet exploited for
cross-scenario modulation classification, here transferred to
cross-receiver RF fingerprinting.
"""

import torch
import torch.nn as nn

from backbones.PatchNet import PatchNet
from backbones.HCA import HermitianCovarianceAttention


class PatchNetHCA(PatchNet):

    def __init__(self, *args, n_bands=8, hca_reduction=2, hca_dropout=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bands = n_bands
        n_modes = self.max_modes  # 2 * max_k - 1
        # pad spectrum up to a multiple of n_bands
        self.pad = (n_bands - n_modes % n_bands) % n_bands
        self.padded_modes = n_modes + self.pad
        self.band_len = self.padded_modes // n_bands
        self.hca = HermitianCovarianceAttention(
            n_bands, reduction=hca_reduction, dropout=hca_dropout
        )

    def forward(self, x, return_k=False, tau=1.0):
        B, C, L = x.shape
        c = self.center

        x_complex = torch.complex(x[:, 0], x[:, 1])
        Xf_shift = torch.fft.fftshift(torch.fft.fft(x_complex, dim=-1), dim=-1)

        Xf_masked, selected_k = self.cutoff(Xf_shift, tau=tau)

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

        x_recon = torch.fft.ifft(torch.fft.ifftshift(Xf_full_shift, dim=-1), dim=-1)
        energy = torch.sum(x_recon.abs().pow(2), dim=-1, keepdim=True)
        x_normalized = x_recon / (torch.sqrt(energy) + 1e-8)

        x_out = torch.stack([x_normalized.real, x_normalized.imag], dim=1)
        tokens = self.mlp_block(self.patch_embed(x_out))
        logits = self.cls_head(tokens.mean(dim=1))

        if return_k:
            return logits, selected_k
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L = 4, 256
    model = PatchNetHCA(L=L, num_classes=6)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(B, 2, L))
    print(f"logits: {tuple(logits.shape)}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
