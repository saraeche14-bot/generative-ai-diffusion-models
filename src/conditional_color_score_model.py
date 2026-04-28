# -*- coding: utf-8 -*-
"""
Conditional score model for colored MNIST.

The model estimates:

    s_theta(x_t, t, digit, color)

where:
- x_t is a noisy RGB image,
- t is the diffusion time,
- digit is the MNIST digit label,
- color is the color label.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features for encoding diffusion time."""

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.rff_weights = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.rff_weights[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """Linear layer reshaped to be added to feature maps."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)[..., None, None]


class ConditionalColorScoreNet(nn.Module):
    """
    Time-, digit- and color-conditional score network for RGB MNIST.

    Inputs:
        x: noisy RGB image, shape (batch_size, 3, 28, 28)
        t: diffusion time, shape (batch_size,)
        digit_label: digit class, shape (batch_size,)
        color_label: color class, shape (batch_size,)

    Output:
        Estimated score with same shape as x.
    """

    def __init__(
        self,
        marginal_prob_std,
        num_digits: int = 10,
        num_colors: int = 7,
        channels: list[int] = [32, 64, 128, 256],
        embed_dim: int = 256,
    ):
        super().__init__()

        self.time_embed = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.digit_embed = nn.Embedding(num_digits, embed_dim)
        self.color_embed = nn.Embedding(num_colors, embed_dim)

        # Encoder: now input has 3 RGB channels
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoder
        self.tconv4 = nn.ConvTranspose2d(
            channels[3],
            channels[2],
            3,
            stride=2,
            bias=False,
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        # Output has 3 RGB channels
        self.tconv1 = nn.ConvTranspose2d(
            channels[0] + channels[0],
            3,
            3,
            stride=1,
        )

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        digit_label: torch.Tensor,
        color_label: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""

        time_embed = self.act(self.time_embed(t))
        digit_embed = self.digit_embed(digit_label)
        color_embed = self.color_embed(color_label)

        embed = time_embed + digit_embed + color_embed

        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.marginal_prob_std(t)[:, None, None, None]

        return h