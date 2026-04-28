# -*- coding: utf-8 -*-
"""
Utilities for Ornstein-Uhlenbeck / variance-preserving diffusion models.

This module contains reusable functions for:
- noise schedules beta(t),
- the forward OU/VP diffusion process,
- the reverse-time drift,
- the construction of the Gaussian diffusion process,
- sample generation with Euler-Maruyama.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch

import diffusion_process as dfp


# Noise schedules

def linear_beta_schedule(
    t: torch.Tensor,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
) -> torch.Tensor:
    """
    Linear noise schedule:
        beta(t) = beta_min + (beta_max - beta_min) * t

    Args:
        t: Diffusion times in [0, 1], shape (batch_size,).
        beta_min: Minimum beta value.
        beta_max: Maximum beta value.

    Returns:
        Tensor of shape (batch_size,) with beta(t).
    """
    return beta_min + (beta_max - beta_min) * t


def linear_beta_integral(
    t: torch.Tensor,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
) -> torch.Tensor:
    """
    Integral of the linear schedule from 0 to t:
        ∫_0^t beta(s) ds = beta_min * t + 0.5 * (beta_max - beta_min) * t^2
    """
    return beta_min * t + 0.5 * (beta_max - beta_min) * t**2


def constant_beta_schedule(
    t: torch.Tensor,
    beta_const: float = 5.0,
) -> torch.Tensor:
    """
    Constant noise schedule:
        beta(t) = beta_const
    """
    return beta_const * torch.ones_like(t)


def constant_beta_integral(
    t: torch.Tensor,
    beta_const: float = 5.0,
) -> torch.Tensor:
    """
    Integral of the constant schedule from 0 to t:
        ∫_0^t beta(s) ds = beta_const * t
    """
    return beta_const * t


def cosine_alpha_bar(
    t: torch.Tensor,
    s: float = 0.008,
) -> torch.Tensor:
    """
    Cosine alpha_bar schedule used in VP diffusion models.

    alpha_bar(t) = cos^2( (pi/2) * (t + s)/(1 + s) ) / cos^2( (pi/2) * s/(1 + s) )

    Args:
        t: Diffusion times in [0, 1], shape (batch_size,).
        s: Small offset used in the cosine schedule.

    Returns:
        Tensor of shape (batch_size,) with alpha_bar(t).
    """
    numerator = torch.cos(0.5 * np.pi * (t + s) / (1.0 + s)) ** 2
    denominator = np.cos(0.5 * np.pi * s / (1.0 + s)) ** 2
    return numerator / denominator


def cosine_beta_schedule(
    t: torch.Tensor,
    s: float = 0.008,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """
    Continuous beta(t) induced by the cosine alpha_bar schedule.

    Since alpha_bar(t) = exp(-∫_0^t beta(u) du), then:
        beta(t) = - d/dt log(alpha_bar(t))

    For the cosine schedule:
        beta(t) = (pi / (1 + s)) * tan( (pi/2) * (t + s)/(1 + s) )

    Args:
        t: Diffusion times in [0, 1], shape (batch_size,).
        s: Offset parameter.
        eps: Small numerical clamp.

    Returns:
        Tensor of shape (batch_size,) with beta(t).
    """
    angle = 0.5 * np.pi * (t + s) / (1.0 + s)
    beta_t = (np.pi / (1.0 + s)) * torch.tan(angle)
    return torch.clamp(beta_t, min=eps, max=100.0)


def cosine_beta_integral(
    t: torch.Tensor,
    s: float = 0.008,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """
    Integral of beta(t) obtained from alpha_bar:
        ∫_0^t beta(u) du = -log(alpha_bar(t))

    Args:
        t: Diffusion times in [0, 1], shape (batch_size,).
        s: Offset parameter.
        eps: Clamp to avoid log(0).

    Returns:
        Tensor of shape (batch_size,).
    """
    alpha_bar_t = torch.clamp(cosine_alpha_bar(t, s=s), min=eps)
    return -torch.log(alpha_bar_t)


# OU / VP diffusion process

def ou_drift_coefficient(
    x_t: torch.Tensor,
    t: torch.Tensor,
    beta_schedule,
) -> torch.Tensor:
    """
    Drift coefficient of the OU/VP forward SDE:

        dx(t) = -0.5 * beta(t) * x(t) dt + sqrt(beta(t)) dW(t)
    """
    beta_t = beta_schedule(t)[:, None, None, None]
    return -0.5 * beta_t * x_t


def ou_diffusion_coefficient(
    t: torch.Tensor,
    beta_schedule,
) -> torch.Tensor:
    """
    Diffusion coefficient of the OU/VP forward SDE:
        g(t) = sqrt(beta(t))
    """
    return torch.sqrt(beta_schedule(t))


def ou_alpha_t(
    t: torch.Tensor,
    beta_integral,
) -> torch.Tensor:
    """
    alpha(t) = exp( -0.5 * ∫_0^t beta(s) ds )
    """
    return torch.exp(-0.5 * beta_integral(t))


def ou_mu_t(
    x_0: torch.Tensor,
    t: torch.Tensor,
    beta_integral,
) -> torch.Tensor:
    """
    Conditional mean of x(t) given x(0)=x_0:
        mu_t = alpha(t) * x_0
    """
    alpha_t = ou_alpha_t(t, beta_integral=beta_integral)
    return alpha_t[:, None, None, None] * x_0


def ou_sigma_t(
    t: torch.Tensor,
    beta_integral,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """
    Conditional standard deviation of x(t) given x(0)=x_0:
        sigma_t = sqrt(1 - alpha(t)^2)
    """
    alpha_t = ou_alpha_t(t, beta_integral=beta_integral)
    sigma2_t = torch.clamp(1.0 - alpha_t**2, min=eps)
    return torch.sqrt(sigma2_t)


def build_ou_diffusion_process(
    beta_schedule,
    beta_integral,
) -> dfp.GaussianDiffussionProcess:
    """
    Build a GaussianDiffussionProcess instance for the OU/VP model.
    """
    drift_coefficient = partial(
        ou_drift_coefficient,
        beta_schedule=beta_schedule,
    )
    diffusion_coefficient = partial(
        ou_diffusion_coefficient,
        beta_schedule=beta_schedule,
    )
    mu_t = partial(
        ou_mu_t,
        beta_integral=beta_integral,
    )
    sigma_t = partial(
        ou_sigma_t,
        beta_integral=beta_integral,
    )

    return dfp.GaussianDiffussionProcess(
        drift_coefficient,
        diffusion_coefficient,
        mu_t,
        sigma_t,
    )


# Reverse-time generation

def ou_backward_drift_coefficient(
    x_t: torch.Tensor,
    t: torch.Tensor,
    score_model,
    beta_schedule,
) -> torch.Tensor:
    """
    Reverse-time drift for the OU/VP model:

        dx(t) = [ -0.5 beta(t) x(t) - beta(t) score(x,t) ] dt + sqrt(beta(t)) dW_bar(t)
    """
    beta_t = beta_schedule(t)[:, None, None, None]
    return -0.5 * beta_t * x_t - beta_t * score_model(x_t, t)


def sample_ou_initial_noise(
    n_images: int,
    image_shape: tuple[int, int, int],
    device: str | torch.device,
) -> torch.Tensor:
    """
    Sample the initial condition x(T) ~ N(0, I) for reverse-time generation.
    """
    return torch.randn(n_images, *image_shape, device=device)


def sample_ou_euler_maruyama(
    score_model,
    beta_schedule,
    n_images: int,
    image_shape: tuple[int, int, int] = (1, 28, 28),
    T: float = 1.0,
    t_end: float = 1.0e-3,
    n_steps: int = 500,
    device: str | torch.device = "cpu",
):
    """
    Generate samples by integrating the reverse-time OU/VP SDE with Euler-Maruyama.
    """
    diffusion_coefficient = partial(
        ou_diffusion_coefficient,
        beta_schedule=beta_schedule,
    )

    image_T = sample_ou_initial_noise(
        n_images=n_images,
        image_shape=image_shape,
        device=device,
    )

    with torch.no_grad():
        times, synthetic_images_t = dfp.euler_maruyama_integrator(
            image_T,
            t_0=T,
            t_end=t_end,
            n_steps=n_steps,
            drift_coefficient=partial(
                ou_backward_drift_coefficient,
                score_model=score_model,
                beta_schedule=beta_schedule,
            ),
            diffusion_coefficient=diffusion_coefficient,
        )

    return times, synthetic_images_t



def langevin_corrector(
    x_t: torch.Tensor,
    t: torch.Tensor,
    score_model,
    snr: float = 0.16,
    n_corrector_steps: int = 1,
):
    """
    Apply Langevin correction steps at a fixed diffusion time t.
    """
    for _ in range(n_corrector_steps):
        grad = score_model(x_t, t)
        noise = torch.randn_like(x_t)

        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=1).mean()

        step_size = 2.0 * (snr * noise_norm / (grad_norm + 1.0e-12)) ** 2

        x_t = x_t + step_size * grad + torch.sqrt(2.0 * step_size) * noise

    return x_t


def sample_ou_predictor_corrector(
    score_model,
    beta_schedule,
    n_images: int,
    image_shape: tuple[int, int, int] = (1, 28, 28),
    T: float = 1.0,
    t_end: float = 1.0e-3,
    n_steps: int = 500,
    n_corrector_steps: int = 1,
    snr: float = 0.16,
    device: str | torch.device = "cpu",
):
    """
    Generate samples with a predictor-corrector sampler for the OU/VP model.
    """
    diffusion_coefficient = partial(
        ou_diffusion_coefficient,
        beta_schedule=beta_schedule,
    )

    times = torch.linspace(T, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x_t = torch.empty(
        (n_images, *image_shape, n_steps + 1),
        dtype=torch.float32,
        device=device,
    )

    x_current = sample_ou_initial_noise(
        n_images=n_images,
        image_shape=image_shape,
        device=device,
    )
    x_t[..., 0] = x_current

    with torch.no_grad():
        for n, t_scalar in enumerate(times[:-1]):
            t_batch = torch.ones(n_images, device=device) * t_scalar

            # Corrector
            x_current = langevin_corrector(
                x_t=x_current,
                t=t_batch,
                score_model=score_model,
                snr=snr,
                n_corrector_steps=n_corrector_steps,
            )

            # Predictor
            drift = ou_backward_drift_coefficient(
                x_t=x_current,
                t=t_batch,
                score_model=score_model,
                beta_schedule=beta_schedule,
            )

            g_t = diffusion_coefficient(t_batch)[:, None, None, None]
            noise = torch.randn_like(x_current)

            x_current = x_current + drift * dt + g_t * torch.sqrt(torch.abs(dt)) * noise
            x_t[..., n + 1] = x_current

    return times, x_t


def ou_probability_flow_drift_coefficient(
    x_t: torch.Tensor,
    t: torch.Tensor,
    score_model,
    beta_schedule,
) -> torch.Tensor:
    """
    Drift of the probability flow ODE for the OU/VP model.

    Reverse probability flow ODE:
        dx/dt = -0.5 * beta(t) * x - 0.5 * beta(t) * score(x,t)
    """
    beta_t = beta_schedule(t)[:, None, None, None]
    return -0.5 * beta_t * x_t - 0.5 * beta_t * score_model(x_t, t)


def ode_integrator(
    x_0: torch.Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient,
):
    """
    Deterministic ODE integrator based on explicit Euler.
    """
    device = x_0.device
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x_t = torch.empty(
        (*x_0.shape, len(times)),
        dtype=x_0.dtype,
        device=device,
    )
    x_t[..., 0] = x_0

    for n, t_scalar in enumerate(times[:-1]):
        t_batch = torch.ones(x_0.shape[0], device=device) * t_scalar
        x_t[..., n + 1] = x_t[..., n] + drift_coefficient(x_t[..., n], t_batch) * dt

    return times, x_t


def sample_ou_probability_flow_ode(
    score_model,
    beta_schedule,
    n_images: int,
    image_shape: tuple[int, int, int] = (1, 28, 28),
    T: float = 1.0,
    t_end: float = 1.0e-3,
    n_steps: int = 500,
    device: str | torch.device = "cpu",
):
    """
    Generate samples by integrating the probability flow ODE
    for the OU/VP model.
    """
    image_T = sample_ou_initial_noise(
        n_images=n_images,
        image_shape=image_shape,
        device=device,
    )

    with torch.no_grad():
        times, synthetic_images_t = ode_integrator(
            image_T,
            t_0=T,
            t_end=t_end,
            n_steps=n_steps,
            drift_coefficient=partial(
                ou_probability_flow_drift_coefficient,
                score_model=score_model,
                beta_schedule=beta_schedule,
            ),
        )

    return times, synthetic_images_t