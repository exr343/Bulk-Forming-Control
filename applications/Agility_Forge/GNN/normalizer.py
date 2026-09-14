"""Online (running-statistics) feature normalizer.

PyTorch port of the vendored reference's `normalization.py` (per-component
mean/std accumulated online during training, frozen after
`max_accumulations` calls) -- not a precompute-once scale the way the
koopman module's pooled min-max normalization works. See GNN/README.md's
"Normalization" section for why this convention was deliberately chosen over
koopman's, and the accepted tradeoffs (moving-target early-training effect,
per-axis rather than pooled/direction-preserving scaling).
"""

import torch
import torch.nn as nn


class OnlineNormalizer(nn.Module):
    """Normalizes the last dimension of its input to zero mean / unit variance.

    Statistics are accumulated per-component (independently for each of the
    `size` features) across every call made with `accumulate=True` while the
    module is in training mode, up to `max_accumulations` calls -- exactly
    the reference `Normalizer`'s semantics, ported from TF variables to
    PyTorch buffers (so they move with the model via `.to(device)` and are
    saved/restored by `state_dict()`/`load_state_dict()` automatically).
    """

    def __init__(self, size: int, max_accumulations: int = 10**6, std_epsilon: float = 1e-8):
        super().__init__()
        self.max_accumulations = max_accumulations
        self.std_epsilon = std_epsilon
        self.register_buffer("acc_count", torch.zeros(()))
        self.register_buffer("num_accumulations", torch.zeros(()))
        self.register_buffer("acc_sum", torch.zeros(size))
        self.register_buffer("acc_sum_squared", torch.zeros(size))

    def _mean(self) -> torch.Tensor:
        safe_count = torch.clamp(self.acc_count, min=1.0)
        return self.acc_sum / safe_count

    def _std(self) -> torch.Tensor:
        safe_count = torch.clamp(self.acc_count, min=1.0)
        # Clamp the variance at 0 before sqrt: floating-point error in
        # sum_sq/count - mean**2 can go slightly negative near zero true
        # variance, which would otherwise NaN (the reference's tf.sqrt has
        # the same latent issue -- this is a defensive fix, not a behavior
        # change on well-conditioned data).
        var = self.acc_sum_squared / safe_count - self._mean() ** 2
        return torch.clamp(var, min=0.0).sqrt().clamp(min=self.std_epsilon)

    @torch.no_grad()
    def _accumulate(self, x_flat: torch.Tensor):
        self.acc_sum += x_flat.sum(dim=0)
        self.acc_sum_squared += (x_flat**2).sum(dim=0)
        self.acc_count += x_flat.shape[0]
        self.num_accumulations += 1

    def forward(self, x: torch.Tensor, accumulate: bool = True) -> torch.Tensor:
        """Normalizes `x` (..., size). Accumulates stats iff `accumulate` and
        `self.training` and the accumulation cap hasn't been reached yet --
        matching the reference's `is_training`-gated accumulation exactly."""
        if accumulate and self.training and self.num_accumulations.item() < self.max_accumulations:
            self._accumulate(x.reshape(-1, x.shape[-1]))
        return (x - self._mean()) / self._std()

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._std() + self._mean()
