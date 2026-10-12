"""Robust loss terms for SMPL fitting."""
from __future__ import annotations

from typing import Any


def charbonnier(x: Any, eps: float = 1e-3) -> Any:
    """Charbonnier robust penalty."""
    return (x * x + eps * eps).sqrt()


def robust_clamped(x: Any, clamp: float, eps: float = 1e-3) -> Any:
    """Clamp residuals before Charbonnier penalty."""
    return charbonnier(x.clamp(min=-clamp, max=clamp), eps=eps)


def masked_mean(values: Any, mask: Any) -> Any:
    """Mean over mask, returning a differentiable zero if empty."""
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def skin_depth_loss(p_model: Any, p_obs: Any, n_obs: Any, mask: Any, residual_clamp: float) -> Any:
    """Two-sided skin point-to-plane loss plus weak Z residual."""
    r_plane = (n_obs * (p_model - p_obs)).sum(dim=-1)
    r_z = p_model[:, 2] - p_obs[:, 2]
    return masked_mean(robust_clamped(r_plane, residual_clamp) + 0.2 * robust_clamped(r_z, residual_clamp), mask)


def cloth_inside_loss(z_model: Any, z_obs: Any, mask: Any, residual_clamp: float, margin_m: float) -> Any:
    """One-sided cloth outer-surface constraint.

    Camera Z grows away from camera. Body should not be closer than clothing
    depth plus a tiny margin.
    """
    r = ((z_obs + margin_m) - z_model).clamp(min=0.0)
    return masked_mean(robust_clamped(r, residual_clamp), mask)

