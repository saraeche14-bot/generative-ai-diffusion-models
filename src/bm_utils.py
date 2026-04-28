# -*- coding: utf-8 -*-
"""
Utilities for Brownian-motion-based diffusion models (VE).

This module contains reusable functions for:
- the forward diffusion process,
- the reverse-time drift,
- the construction of the Gaussian diffusion process,
- sample generation with Euler-Maruyama.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch

import diffusion_process as dfp


def bm_drift_coefficient(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Drift coefficient of the Brownian motion forward SDE.

    Forward SDE:
        dx(t) = sigma^t dW(t)

    Therefore, the drift term is zero.

    Args:
        x_t: Noisy images at time t.
        t: Diffusion times.

    Returns:
        Tensor with the same shape as x_t containing zeros.
    """
    return torch.zeros_like(x_t)


def bm_diffusion_coefficient(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Diffusion coefficient g(t) = sigma^t of the Brownian motion forward SDE.

    Args:
        t: Diffusion times, shape (batch_size,).
        sigma: Positive scalar controlling the noise growth.

    Returns:
        Tensor of shape (batch_size,) with the diffusion coefficient at each time.
    """
    return sigma ** t


def bm_mu_t(x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Conditional mean of x(t) given x(0)=x_0 for Brownian motion.

    Since the drift is zero, the conditional mean is constant:
        mu_t(x_0, t) = x_0

    Args:
        x_0: Clean images.
        t: Diffusion times.

    Returns:
        Conditional mean with the same shape as x_0.
    """
    return x_0


def bm_sigma_t(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Conditional standard deviation of x(t) given x(0)=x_0 for Brownian motion.

    For g(t)=sigma^t:
        sigma_t^2 = integral_0^t sigma^(2s) ds
                  = (sigma^(2t) - 1) / (2 log(sigma))

    Args:
        t: Diffusion times, shape (batch_size,).
        sigma: Positive scalar controlling the noise growth.

    Returns:
        Tensor of shape (batch_size,) with the conditional standard deviation.
    """
    return torch.sqrt(0.5 * (sigma ** (2 * t) - 1.0) / np.log(sigma))


def build_bm_diffusion_process(sigma: float) -> dfp.GaussianDiffussionProcess:
    """
    Build a GaussianDiffussionProcess instance for the Brownian motion model.

    Args:
        sigma: Positive scalar controlling the variance-exploding diffusion.

    Returns:
        Configured GaussianDiffussionProcess object.
    """
    drift_coefficient = bm_drift_coefficient
    diffusion_coefficient = partial(bm_diffusion_coefficient, sigma=sigma)
    mu_t = bm_mu_t
    sigma_t = partial(bm_sigma_t, sigma=sigma)

    return dfp.GaussianDiffussionProcess(
        drift_coefficient,
        diffusion_coefficient,
        mu_t,
        sigma_t,
    )


def backward_drift_coefficient(
    x_t: torch.Tensor,
    t: torch.Tensor,
    score_model,
    diffusion_coefficient,
) -> torch.Tensor:
    """
    Drift of the reverse-time SDE for the Brownian motion model.

    Reverse SDE:
        dx(t) = -g(t)^2 score(x,t) dt + g(t) dW_bar(t)

    Args:
        x_t: Current noisy images.
        t: Diffusion times.
        score_model: Trained score network.
        diffusion_coefficient: Function g(t).

    Returns:
        Reverse-time drift tensor with the same shape as x_t.
    """
    g_t = diffusion_coefficient(t)[:, None, None, None]
    return -(g_t ** 2) * score_model(x_t, t)


def sample_initial_noise(
    n_images: int,
    image_shape: tuple[int, int, int],
    sigma: float,
    T: float,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Sample the initial condition x(T) for reverse-time generation.

    We use:
        x(T) ~ N(0, sigma_T^2 I)

    Args:
        n_images: Number of images to generate.
        image_shape: Tuple (channels, height, width).
        sigma: Brownian motion parameter.
        T: Final diffusion time.
        device: Torch device.

    Returns:
        Tensor of shape (n_images, channels, height, width).
    """
    t_tensor = torch.ones(n_images, device=device) * T
    std = bm_sigma_t(t_tensor, sigma=sigma)[:, None, None, None]
    z = torch.randn(n_images, *image_shape, device=device)
    return std * z


def sample_bm_euler_maruyama(
    score_model,
    sigma: float,
    n_images: int,
    image_shape: tuple[int, int, int] = (1, 28, 28),
    T: float = 1.0,
    t_end: float = 1.0e-3,
    n_steps: int = 500,
    device: str | torch.device = "cpu",
):
    """
    Generate samples by integrating the reverse-time SDE with Euler-Maruyama.

    Args:
        score_model: Trained score network.
        sigma: Brownian motion parameter.
        n_images: Number of images to generate.
        image_shape: Tuple (channels, height, width).
        T: Initial reverse-time integration time.
        t_end: Final integration time, close to zero.
        n_steps: Number of Euler-Maruyama steps.
        device: Torch device.

    Returns:
        times: Time grid used in the integration.
        synthetic_images_t: Generated trajectories with shape
            (n_images, channels, height, width, n_steps + 1).
    """
    diffusion_coefficient = partial(bm_diffusion_coefficient, sigma=sigma)

    image_T = sample_initial_noise(
        n_images=n_images,
        image_shape=image_shape,
        sigma=sigma,
        T=T,
        device=device,
    )

    with torch.no_grad():
        times, synthetic_images_t = dfp.euler_maruyama_integrator(
            image_T,
            t_0=T,
            t_end=t_end,
            n_steps=n_steps,
            drift_coefficient=partial(
                backward_drift_coefficient,
                score_model=score_model,
                diffusion_coefficient=diffusion_coefficient,
            ),
            diffusion_coefficient=diffusion_coefficient,
        )

    return times, synthetic_images_t



def probability_flow_drift_coefficient(
    x_t: torch.Tensor,
    t: torch.Tensor,
    score_model,
    diffusion_coefficient,
) -> torch.Tensor:
    """
    Drift of the probability flow ODE for the Brownian motion model.

    Probability flow ODE:
        dx(t)/dt = -0.5 * g(t)^2 * score(x,t)

    Args:
        x_t: Current noisy images.
        t: Diffusion times.
        score_model: Trained score network.
        diffusion_coefficient: Function g(t).

    Returns:
        Drift tensor with the same shape as x_t.
    """
    g_t = diffusion_coefficient(t)[:, None, None, None]
    return -0.5 * (g_t ** 2) * score_model(x_t, t)


def ode_integrator(
    x_0: torch.Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient,
):
    """
    Deterministic ODE integrator based on the explicit Euler method.

    Args:
        x_0: Initial images of shape
            (batch_size, n_channels, image_height, image_width).
        t_0: Initial integration time.
        t_end: Final integration time.
        n_steps: Number of integration steps.
        drift_coefficient: Function of (x(t), t) defining the ODE drift.

    Returns:
        times: Time grid.
        x_t: Deterministic trajectories with shape
            (*np.shape(x_0), n_steps + 1).
    """
    device = x_0.device

    # Create the time grid
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    # Allocate memory for the trajectories
    x_t = torch.empty(
        (*x_0.shape, len(times)),
        dtype=x_0.dtype,
        device=device,
    )
    x_t[..., 0] = x_0

    # Euler integration of the ODE
    for n, t in enumerate(times[:-1]):
        t_batch = torch.ones(x_0.shape[0], device=device) * t
        x_t[..., n + 1] = x_t[..., n] + drift_coefficient(x_t[..., n], t_batch) * dt

    return times, x_t


def sample_bm_probability_flow_ode(
    score_model,
    sigma: float,
    n_images: int,
    image_shape: tuple[int, int, int] = (1, 28, 28),
    T: float = 1.0,
    t_end: float = 1.0e-3,
    n_steps: int = 500,
    device: str | torch.device = "cpu",
):
    """
    Generate samples by integrating the probability flow ODE
    for the Brownian motion model.

    Args:
        score_model: Trained score network.
        sigma: Brownian motion parameter.
        n_images: Number of images to generate.
        image_shape: Tuple (channels, height, width).
        T: Initial reverse-time integration time.
        t_end: Final integration time, close to zero.
        n_steps: Number of integration steps.
        device: Torch device.

    Returns:
        times: Time grid.
        synthetic_images_t: Generated trajectories with shape
            (n_images, channels, height, width, n_steps + 1).
    """
    diffusion_coefficient = partial(bm_diffusion_coefficient, sigma=sigma)

    image_T = sample_initial_noise(
        n_images=n_images,
        image_shape=image_shape,
        sigma=sigma,
        T=T,
        device=device,
    )

    with torch.no_grad():
        times, synthetic_images_t = ode_integrator(
            image_T,
            t_0=T,
            t_end=t_end,
            n_steps=n_steps,
            drift_coefficient=partial(
                probability_flow_drift_coefficient,
                score_model=score_model,
                diffusion_coefficient=diffusion_coefficient,
            ),
        )

    return times, synthetic_images_t



def gaussian_log_density(
    x: torch.Tensor,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    """
    Log-density of a Gaussian N(0, sigma_t^2 I) evaluated at x.

    Args:
        x: Tensor of shape (batch_size, channels, height, width).
        sigma_t: Tensor of shape (batch_size,) with the standard deviation.

    Returns:
        Tensor of shape (batch_size,) with the Gaussian log-density.
    """
    batch_size = x.shape[0]
    dim = x[0].numel()

    x_flat = x.view(batch_size, -1)
    sigma2 = sigma_t ** 2

    return (
        -0.5 * dim * torch.log(2.0 * torch.pi * sigma2)
        - 0.5 * torch.sum(x_flat ** 2, dim=1) / sigma2
    )


def probability_flow_divergence(
    x_t: torch.Tensor,
    t: torch.Tensor,
    score_model,
    diffusion_coefficient,
) -> torch.Tensor:
    """
    Exact divergence of the probability flow drift with respect to x.

    Drift:
        v(x,t) = -0.5 * g(t)^2 * score_model(x,t)

    Args:
        x_t: Current batch of images, shape (batch_size, C, H, W).
        t: Diffusion times, shape (batch_size,).
        score_model: Trained score network.
        diffusion_coefficient: Function g(t).

    Returns:
        Tensor of shape (batch_size,) containing div_x v(x,t).
    """
    x_t = x_t.requires_grad_(True)

    drift = probability_flow_drift_coefficient(
        x_t=x_t,
        t=t,
        score_model=score_model,
        diffusion_coefficient=diffusion_coefficient,
    )

    batch_size = x_t.shape[0]
    dim = x_t[0].numel()

    drift_flat = drift.view(batch_size, -1)
    x_flat = x_t.view(batch_size, -1)

    divergence = torch.zeros(batch_size, device=x_t.device)

    for i in range(dim):
        grad_i = torch.autograd.grad(
            outputs=drift_flat[:, i].sum(),
            inputs=x_t,
            create_graph=False,
            retain_graph=True,
        )[0].view(batch_size, -1)[:, i]
        divergence += grad_i

    return divergence

def compute_log_likelihood_ode(
    x_0: torch.Tensor,
    score_model,
    sigma: float,
    T: float = 1.0,
    n_steps: int = 200,
    eps: float = 1.0e-3,
):
    """
    Estimate log p_0(x_0) using the probability flow ODE.

    The method integrates forward in time from t=eps to t=T:
        dx/dt = -0.5 * g(t)^2 * score(x,t)

    and simultaneously accumulates:
        d log p_t(x(t)) / dt = - div_x v(x(t), t)

    Therefore,
        log p_eps(x(eps)) = log p_T(x_T) + integral_eps^T div_x v(x(t), t) dt

    Args:
        x_0: Batch of real images, shape (batch_size, C, H, W).
        score_model: Trained score network.
        sigma: Brownian motion parameter.
        T: Final time.
        n_steps: Number of Euler steps.
        eps: Small positive initial time to avoid numerical instability at t=0.

    Returns:
        log_p_0: Estimated log-likelihoods, shape (batch_size,)
        times: Time grid
        x_t: ODE trajectories
    """
    device = x_0.device
    diffusion_coefficient = partial(bm_diffusion_coefficient, sigma=sigma)

    times = torch.linspace(eps, T, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x_t = torch.empty(
        (*x_0.shape, len(times)),
        dtype=x_0.dtype,
        device=device,
    )
    x_t[..., 0] = x_0

    log_det_correction = torch.zeros(x_0.shape[0], device=device)

    for n, t in enumerate(times[:-1]):
        t_batch = torch.ones(x_0.shape[0], device=device) * t

        x_current = x_t[..., n]

        drift = probability_flow_drift_coefficient(
            x_t=x_current,
            t=t_batch,
            score_model=score_model,
            diffusion_coefficient=diffusion_coefficient,
        )

        divergence = probability_flow_divergence(
            x_t=x_current,
            t=t_batch,
            score_model=score_model,
            diffusion_coefficient=diffusion_coefficient,
        )

        x_t[..., n + 1] = x_current + drift * dt
        log_det_correction += divergence * dt

    sigma_T = bm_sigma_t(torch.ones(x_0.shape[0], device=device) * T, sigma=sigma)
    log_p_T = gaussian_log_density(x_t[..., -1], sigma_T)

    log_p_0 = log_p_T + log_det_correction

    return log_p_0, times, x_t


def langevin_corrector(
    x_t: torch.Tensor,
    t: torch.Tensor,
    score_model,
    snr: float = 0.16,
    n_corrector_steps: int = 1,
):
    """
    Apply Langevin correction steps at a fixed diffusion time t.

    Args:
        x_t: Current noisy images, shape (batch_size, C, H, W).
        t: Diffusion times, shape (batch_size,).
        score_model: Trained score network.
        snr: Signal-to-noise ratio controlling the Langevin step size.
        n_corrector_steps: Number of correction steps.

    Returns:
        Refined batch of images after Langevin correction.
    """
    for _ in range(n_corrector_steps):
        grad = score_model(x_t, t)
        noise = torch.randn_like(x_t)

        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=1).mean()

        step_size = 2.0 * (snr * noise_norm / (grad_norm + 1.0e-12)) ** 2

        x_t = x_t + step_size * grad + torch.sqrt(2.0 * step_size) * noise

    return x_t


def sample_bm_predictor_corrector(
    score_model,
    sigma: float,
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
    Generate samples with a predictor-corrector sampler for the BM/VE model.

    Predictor:
        One reverse-time Euler-Maruyama step.

    Corrector:
        One or more Langevin correction steps.

    Returns:
        times: Time grid.
        synthetic_images_t: Trajectories of shape
            (n_images, channels, height, width, n_steps + 1).
    """
    diffusion_coefficient = partial(bm_diffusion_coefficient, sigma=sigma)

    times = torch.linspace(T, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x_t = torch.empty(
        (n_images, *image_shape, n_steps + 1),
        dtype=torch.float32,
        device=device,
    )

    x_current = sample_initial_noise(
        n_images=n_images,
        image_shape=image_shape,
        sigma=sigma,
        T=T,
        device=device,
    )
    x_t[..., 0] = x_current

    with torch.no_grad():
        for n, t_scalar in enumerate(times[:-1]):
            t_batch = torch.ones(n_images, device=device) * t_scalar

            # -------------------
            # Corrector
            # -------------------
            x_current = langevin_corrector(
                x_t=x_current,
                t=t_batch,
                score_model=score_model,
                snr=snr,
                n_corrector_steps=n_corrector_steps,
            )

            # -------------------
            # Predictor
            # -------------------
            drift = backward_drift_coefficient(
                x_t=x_current,
                t=t_batch,
                score_model=score_model,
                diffusion_coefficient=diffusion_coefficient,
            )

            g_t = diffusion_coefficient(t_batch)[:, None, None, None]
            noise = torch.randn_like(x_current)

            x_current = x_current + drift * dt + g_t * torch.sqrt(torch.abs(dt)) * noise
            x_t[..., n + 1] = x_current

    return times, x_t