# -*- coding: utf-8 -*-
"""
Training utilities for digit- and color-conditional diffusion models.
"""

from __future__ import annotations

import torch


def conditional_color_loss_function(
    diffusion_process,
    score_model,
    x_0: torch.Tensor,
    digit_label: torch.Tensor,
    color_label: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """
    Conditional denoising score-matching loss for colored MNIST.

    The model learns:

        s_theta(x_t, t, digit, color)

    Args:
        diffusion_process:
            Gaussian diffusion process with methods mu_t(x_0, t) and sigma_t(t).
        score_model:
            ConditionalColorScoreNet.
        x_0:
            Clean RGB images with shape (batch_size, 3, 28, 28).
        digit_label:
            Digit labels with shape (batch_size,).
        color_label:
            Color labels with shape (batch_size,).
        eps:
            Small value to avoid t = 0.

    Returns:
        Scalar loss tensor.
    """

    t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps

    z = torch.randn_like(x_0)

    mu = diffusion_process.mu_t(x_0, t)
    sigma = diffusion_process.sigma_t(t)[:, None, None, None]

    x_t = mu + sigma * z

    score = score_model(
        x_t,
        t,
        digit_label,
        color_label,
    )

    loss = torch.mean(
        torch.sum(
            (score * sigma + z) ** 2,
            dim=(1, 2, 3),
        )
    )

    return loss