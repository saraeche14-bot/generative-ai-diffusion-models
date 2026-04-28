# -*- coding: utf-8 -*-
"""
Utilities for controllable generation: imputation with OU diffusion models.

This module implements helper functions for:
- building masks,
- masking observed images,
- simulating noisy observations at a given diffusion time,
- imputing missing pixels with the reverse OU SDE using Euler-Maruyama.

"""

from __future__ import annotations

import numpy as np
import torch


def make_right_half_mask(x: torch.Tensor) -> torch.Tensor:
    """
    Create a mask that keeps the left half of the image and hides the right half.

    Args:
        x: Input batch of images with shape (batch, channels, height, width).

    Returns:
        mask: Tensor with the same shape as x.
              Value 1.0 means known pixel.
              Value 0.0 means missing pixel.
    """
    mask = torch.ones_like(x)
    _, _, _, width = x.shape
    mask[:, :, :, width // 2:] = 0.0
    return mask


def make_center_square_mask(x: torch.Tensor, square_size: int = 10) -> torch.Tensor:
    """
    Create a mask that hides a square centered in the image.

    Args:
        x: Input batch of images with shape (batch, channels, height, width).
        square_size: Side length of the square to hide.

    Returns:
        mask: Tensor with the same shape as x.
    """
    mask = torch.ones_like(x)
    _, _, height, width = x.shape

    h0 = (height - square_size) // 2
    h1 = h0 + square_size
    w0 = (width - square_size) // 2
    w1 = w0 + square_size

    mask[:, :, h0:h1, w0:w1] = 0.0
    return mask


def apply_mask(x: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """
    Apply a binary mask to an image batch.

    Args:
        x: Input images.
        mask: Binary mask with same shape as x.
        fill_value: Value used for missing pixels.

    Returns:
        Masked images.
    """
    return mask * x + (1.0 - mask) * fill_value


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """
    Continuous-time cosine schedule alpha_bar(t), with t in [0, 1].

    Args:
        t: Time tensor in [0, 1].
        s: Small offset used in the cosine schedule.

    Returns:
        alpha_bar(t).
    """
    numerator = torch.cos(((t + s) / (1.0 + s)) * (np.pi / 2.0)) ** 2
    denominator = np.cos((s / (1.0 + s)) * (np.pi / 2.0)) ** 2
    return numerator / denominator


def cosine_beta_t(t: torch.Tensor, s: float = 0.008, beta_max: float = 20.0) -> torch.Tensor:
    """
    Continuous-time beta(t) induced by the cosine alpha_bar schedule.

    A maximum value is used for numerical stability in reverse-time sampling.
    """
    t = torch.clamp(t, min=1.0e-5, max=0.999)
    u = (t + s) / (1.0 + s)
    beta = (np.pi / (1.0 + s)) * torch.tan((np.pi / 2.0) * u)
    beta = torch.clamp(beta, min=1.0e-5, max=beta_max)
    return beta


def ou_cosine_drift_coefficient(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Forward OU drift:
        f(x,t) = -0.5 * beta(t) * x
    """
    beta_t = cosine_beta_t(t)[:, None, None, None]
    return -0.5 * beta_t * x_t


def ou_cosine_diffusion_coefficient(t: torch.Tensor) -> torch.Tensor:
    """
    Forward OU diffusion coefficient:
        g(t) = sqrt(beta(t))
    """
    beta_t = cosine_beta_t(t)
    return torch.sqrt(beta_t)


def ou_cosine_mu_t(x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Mean of p(x_t | x_0) for the OU cosine model:
        mu_t(x_0) = sqrt(alpha_bar(t)) * x_0
    """
    alpha_bar_t = cosine_alpha_bar(t)[:, None, None, None]
    return torch.sqrt(alpha_bar_t) * x_0


def ou_cosine_sigma_t(t: torch.Tensor) -> torch.Tensor:
    """
    Standard deviation of p(x_t | x_0) for the OU cosine model:
        sigma_t = sqrt(1 - alpha_bar(t))
    """
    alpha_bar_t = cosine_alpha_bar(t)
    alpha_bar_t = torch.clamp(alpha_bar_t, min=1.0e-5, max=1.0)
    return torch.sqrt(1.0 - alpha_bar_t)


def diffuse_observation_at_time(
    x_0: torch.Tensor,
    t: torch.Tensor,
    mu_t_fn,
    sigma_t_fn,
) -> torch.Tensor:
    """
    Generate a noisy observation x_t ~ p(x_t | x_0) for a given batch of times.

    Args:
        x_0: Clean observed images.
        t: Time tensor of shape (batch,).
        mu_t_fn: Function for the conditional mean.
        sigma_t_fn: Function for the conditional std.

    Returns:
        Noisy version of the observation at time t.
    """
    mu = mu_t_fn(x_0, t)
    sigma = sigma_t_fn(t)[:, None, None, None]
    z = torch.randn_like(x_0)
    return mu + sigma * z


def ou_reverse_drift_coefficient(
    x_t: torch.Tensor,
    t: torch.Tensor,
    score_model,
    diffusion_coefficient_fn,
    drift_coefficient_fn,
) -> torch.Tensor:
    """
    Reverse drift for the OU model:
        f_rev(x,t) = f(x,t) - g(t)^2 s_theta(x,t)

    Note:
        Since the integration is done from T to eps with a negative dt,
        this is the correct drift to use in the Euler-Maruyama loop.
    """
    score = score_model(x_t, t)
    g_t = diffusion_coefficient_fn(t)[:, None, None, None]
    forward_drift = drift_coefficient_fn(x_t, t)
    return forward_drift - (g_t ** 2) * score


def impute_ou_euler_maruyama(
    x_observed: torch.Tensor,
    mask: torch.Tensor,
    score_model,
    drift_coefficient_fn,
    diffusion_coefficient_fn,
    mu_t_fn,
    sigma_t_fn,
    T: float = 1.0,
    eps: float = 1.0e-3,
    n_steps: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Impute missing pixels using the reverse OU SDE and Euler-Maruyama.

    Args:
        x_observed: Partially observed clean images, shape (batch, 1, H, W).
        mask: Binary mask with same shape as x_observed.
        score_model: Trained score network.
        drift_coefficient_fn: Forward OU drift function.
        diffusion_coefficient_fn: Forward OU diffusion coefficient function.
        mu_t_fn: Conditional mean of p(x_t | x_0).
        sigma_t_fn: Conditional std of p(x_t | x_0).
        T: Initial reverse time.
        eps: Final reverse time.
        n_steps: Number of reverse steps.

    Returns:
        times: Tensor of shape (n_steps + 1,)
        x_path: Tensor of shape (batch, 1, H, W, n_steps + 1)
    """
    device = x_observed.device
    batch_size = x_observed.shape[0]

    times = torch.linspace(T, eps, n_steps + 1, device=device)
    dt = times[1] - times[0]
    sqrt_abs_dt = np.sqrt(abs(dt.item()))

    x_path = torch.empty(
        (*x_observed.shape, n_steps + 1),
        dtype=torch.float32,
        device=device,
    )

    # Initial sample at time T from the Gaussian reference
    x = torch.randn_like(x_observed)

    # Impose the known region at time T
    t_init = torch.full((batch_size,), times[0], device=device)
    x_obs_t = diffuse_observation_at_time(
        x_0=x_observed,
        t=t_init,
        mu_t_fn=mu_t_fn,
        sigma_t_fn=sigma_t_fn,
    )
    x = mask * x_obs_t + (1.0 - mask) * x
    x_path[..., 0] = x

    for n, t_scalar in enumerate(times[:-1]):
        t = torch.full((batch_size,), t_scalar, device=device)

        with torch.no_grad():
            drift = ou_reverse_drift_coefficient(
                x_t=x,
                t=t,
                score_model=score_model,
                diffusion_coefficient_fn=diffusion_coefficient_fn,
                drift_coefficient_fn=drift_coefficient_fn,
            )
            g_t = diffusion_coefficient_fn(t)[:, None, None, None]
            z = torch.randn_like(x)

            x = x + drift * dt + g_t * sqrt_abs_dt * z

            t_next = torch.full((batch_size,), times[n + 1], device=device)
            x_obs_t_next = diffuse_observation_at_time(
                x_0=x_observed,
                t=t_next,
                mu_t_fn=mu_t_fn,
                sigma_t_fn=sigma_t_fn,
            )

            # Re-impose the observed region
            x = mask * x_obs_t_next + (1.0 - mask) * x

        x_path[..., n + 1] = x

    return times, x_path