# -*- coding: utf-8 -*-
"""
Simulate Gaussian processes.

@author: <alberto.suarez@uam.es>
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor


def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[float, float],
    diffusion_coefficient: Callable[float],
    seed: Union[int, None] = None,
) -> Tensor:
    """Euler-Maruyama integrator (approximate).

    Args:
        x_0: The initial images of dimensions
            (batch_size, n_channels, image_height, image_width)
        t_0: Initial time.
        t_end: Endpoint of the integration interval.
        n_steps: Number of integration steps.
        drift_coefficient: Function of (x(t), t) that defines the drift term.
        diffusion_coefficient: Function of t that defines the diffusion term.
        seed: Seed for the random number generator.

    Returns:
        times: Discretization times.
        x_t: Trajectories that result from the integration of the SDE.
             The shape is (*np.shape(x_0), n_steps + 1).

    Notes:
        The implementation is fully vectorized except for a loop over time.

    Examples:
        >>> import numpy as np
        >>> drift_coefficient = lambda x_t, t: -x_t
        >>> diffusion_coefficient = lambda t: torch.ones_like(t)
        >>> x_0 = torch.tensor(np.reshape(np.arange(120), (2, 3, 5, 4)))
        >>> t_0, t_end = 0.0, 3.0
        >>> n_steps = 6
        >>> times, x_t = euler_maruyama_integrator(
        ...     x_0, t_0, t_end, n_steps, drift_coefficient, diffusion_coefficient,
        ... )
        >>> print(times)
        tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000])
        >>> print(np.shape(x_t))
        torch.Size([2, 3, 5, 4, 7])
    """

    # Select the device where tensors are stored
    device = x_0.device

    # Build the time grid and compute the step size
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    # Allocate memory for the simulated trajectories
    x_t = torch.empty(
        (*x_0.shape, len(times)),
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_0

 
    z = torch.randn_like(x_t)
    z[..., -1] = 0.0  # No noise injection in the last step

    # Advance the trajectories one Euler-Maruyama step at a time
    for n, t in enumerate(times[:-1]):
        t = torch.ones(x_0.shape[0], device=device) * t
        x_t[..., n + 1] = (
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t) * dt
            + diffusion_coefficient(t).view(-1, 1, 1, 1)
            * torch.sqrt(torch.abs(dt))
            * z[..., n]
        )

    return times, x_t


class DiffussionProcess:
    def __init__(
        self,
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient


class GaussianDiffussionProcess(DiffussionProcess):
    """
    Gaussian diffusion process for which the conditional distribution
    p(x_t | x_0) is Gaussian with mean mu_t(x_0, t) and standard
    deviation sigma_t(t).

    This class stores:
        - the drift coefficient of the forward SDE,
        - the diffusion coefficient of the forward SDE,
        - the closed-form conditional mean mu_t(x_0, t),
        - the closed-form conditional standard deviation sigma_t(t).

    These quantities are enough to:
        1. simulate forward trajectories numerically, and
        2. train a score-based model with denoising score matching.

    Example 1:
        >>> mu, sigma = 1.5, 2.0
        >>> bm = GaussianDiffussionProcess(
        ...     drift_coefficient=lambda x_t, t: mu,
        ...     diffusion_coefficient=lambda t: sigma,
        ...     mu_t=lambda x_0, t: x_0 + mu * t,
        ...     sigma_t=lambda t: np.sqrt(2.0 * t),
        ... )
        >>> print(bm.drift_coefficient(x_t=3.0, t=10.0))
        1.5
        >>> print(bm.diffusion_coefficient(t=10.0))
        2.0
        >>> print(bm.mu_t(x_0=3.0, t=10.0), bm.sigma_t(t=10.0))
        18.0 4.47213595499958
    """

    kind = "Gaussian"

    def __init__(
        self,
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
        mu_t: Callable[float, float] = lambda x_0, t: x_0,
        sigma_t: Callable[float] = lambda t: np.sqrt(t),
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
        self.mu_t = mu_t
        self.sigma_t = sigma_t

    def loss_function(
        self,
        score_model,
        x_0: torch.Tensor,
        eps: float = 1.0e-5,
    ):
        """Loss function for training score-based generative models.

        Args:
            score_model: A PyTorch model instance that represents a
                time-dependent score-based model.
            x_0: A mini-batch of training data.
            eps: A tolerance value for numerical stability.

        Returns:
            Scalar loss value computed as the average denoising
            score-matching objective over the mini-batch.
        """

        # Sample one diffusion time per image
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps

        # Sample Gaussian noise with the same shape as the images
        z = torch.randn_like(x_0)

        # Evaluate the conditional standard deviation sigma_t(t)
        sigma = self.sigma_t(t)

        # Generate noisy samples x_t = mu_t(x_0, t) + sigma_t(t) * z
        x_t = self.mu_t(x_0, t) + sigma[:, None, None, None] * z

        # Predict the score at the noisy samples
        score = score_model(x_t, t)

        # Denoising score matching loss
        loss = torch.mean(
            torch.sum(
                (sigma[:, None, None, None] * score + z) ** 2,
                dim=(1, 2, 3),
            )
        )

        return loss


if __name__ == "__main__":
    import doctest
    doctest.testmod()