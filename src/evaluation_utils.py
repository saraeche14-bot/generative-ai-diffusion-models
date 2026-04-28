# -*- coding: utf-8 -*-
"""
Utilities for Exercise 3: evaluation of diffusion models with BPD.

This module provides helper functions to:
- load trained BM / OU score models from checkpoints,
- build the corresponding diffusion process settings,
- compute log-likelihoods and BPD on MNIST data,
- compare models / schedules,
- compare sampling strategies.

Assumption:
Likelihood computation is currently available for the Brownian-motion model
through the probability flow ODE implemented in bm_utils.py.

For OU, this script includes a likelihood approximation via the OU probability
flow ODE, following the same idea used for the BM model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from score_model import ScoreNet

from bm_utils import (
    bm_sigma_t,
    build_bm_diffusion_process,
    bm_diffusion_coefficient,
    sample_bm_euler_maruyama,
    sample_bm_predictor_corrector,
    sample_bm_probability_flow_ode,
    gaussian_log_density,
)

from ou_utils import (
    linear_beta_schedule,
    linear_beta_integral,
    cosine_beta_schedule,
    cosine_beta_integral,
    constant_beta_schedule,
    constant_beta_integral,
    ou_sigma_t,
    build_ou_diffusion_process,
    sample_ou_euler_maruyama,
    sample_ou_predictor_corrector,
    sample_ou_probability_flow_ode,
)


# Basic config container

@dataclass
class ModelConfig:
    model_type: str              # "bm" or "ou"
    schedule_name: str | None    # None for BM; "linear"/"cosine"/"constant" for OU
    checkpoint_path: str
    sigma: float = 25.0          # only for BM


# Dataset helpers

def get_mnist_digit_loader(
    digit: int = 3,
    train: bool = True,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> DataLoader:
    """
    Load MNIST and keep only one digit class.

    Args:
        digit: Digit to keep.
        train: Whether to use the training split.
        batch_size: DataLoader batch size.
        max_samples: Optional cap on the number of selected images.

    Returns:
        DataLoader over the selected subset.
    """
    data = datasets.MNIST(
        root="data",
        train=train,
        download=True,
        transform=ToTensor(),
    )

    indices_digit = torch.where(data.targets == digit)[0]

    if max_samples is not None:
        indices_digit = indices_digit[:max_samples]

    subset = Subset(data, indices_digit)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=torch.get_num_threads(),
    )


# OU schedule builders

def get_ou_schedule_functions(schedule_name: str):
    """
    Return (beta_schedule, beta_integral) for a given OU schedule name.
    """
    if schedule_name == "linear":
        beta_schedule = partial(linear_beta_schedule, beta_min=0.1, beta_max=20.0)
        beta_integral = partial(linear_beta_integral, beta_min=0.1, beta_max=20.0)
    elif schedule_name == "cosine":
        beta_schedule = partial(cosine_beta_schedule, s=0.008)
        beta_integral = partial(cosine_beta_integral, s=0.008)
    elif schedule_name == "constant":
        beta_schedule = partial(constant_beta_schedule, beta_const=5.0)
        beta_integral = partial(constant_beta_integral, beta_const=5.0)
    else:
        raise ValueError(f"Unknown schedule_name: {schedule_name}")

    return beta_schedule, beta_integral


# Model loading

def build_score_model_bm(
    sigma: float,
    checkpoint_path: str,
    device: str | torch.device,
):
    """
    Build and load a BM score model from checkpoint.
    """
    score_model = torch.nn.DataParallel(
        ScoreNet(
            marginal_prob_std=partial(bm_sigma_t, sigma=sigma)
        )
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    score_model.load_state_dict(state_dict)
    score_model.eval()
    return score_model


def build_score_model_ou(
    schedule_name: str,
    checkpoint_path: str,
    device: str | torch.device,
):
    """
    Build and load an OU score model from checkpoint.
    """
    _, beta_integral = get_ou_schedule_functions(schedule_name)

    score_model = torch.nn.DataParallel(
        ScoreNet(
            marginal_prob_std=partial(ou_sigma_t, beta_integral=beta_integral)
        )
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    score_model.load_state_dict(state_dict)
    score_model.eval()
    return score_model


def load_score_model(
    config: ModelConfig,
    device: str | torch.device,
):
    """
    Generic score-model loader.
    """
    if config.model_type == "bm":
        return build_score_model_bm(
            sigma=config.sigma,
            checkpoint_path=config.checkpoint_path,
            device=device,
        )
    elif config.model_type == "ou":
        return build_score_model_ou(
            schedule_name=config.schedule_name,
            checkpoint_path=config.checkpoint_path,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


# OU probability-flow likelihood

def ou_probability_flow_drift_for_likelihood(
    x_t: torch.Tensor,
    t: torch.Tensor,
    score_model,
    beta_schedule,
) -> torch.Tensor:
    """
    Drift of the OU probability flow ODE used for likelihood estimation:

        dx/dt = -0.5 * beta(t) * x - 0.5 * beta(t) * score(x,t)
    """
    beta_t = beta_schedule(t)[:, None, None, None]
    return -0.5 * beta_t * x_t - 0.5 * beta_t * score_model(x_t, t)


def exact_divergence(
    x_t: torch.Tensor,
    t: torch.Tensor,
    drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Exact divergence of a drift vector field wrt x.

    Args:
        x_t: Tensor of shape (batch_size, C, H, W).
        t: Tensor of shape (batch_size,).
        drift_fn: Function returning the drift at (x_t, t).

    Returns:
        Tensor of shape (batch_size,) with div_x v(x,t).
    """
    x_t = x_t.requires_grad_(True)

    drift = drift_fn(x_t, t)
    batch_size = x_t.shape[0]
    dim = x_t[0].numel()

    drift_flat = drift.view(batch_size, -1)
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


def compute_log_likelihood_ode_ou(
    x_0: torch.Tensor,
    score_model,
    beta_schedule,
    beta_integral,
    T: float = 1.0,
    n_steps: int = 200,
    eps: float = 1.0e-3,
):
    """
    Estimate log p_0(x_0) for the OU/VP model using the probability flow ODE.

    We integrate forward from t=eps to t=T and accumulate the divergence:
        d log p_t(x(t)) / dt = - div_x v(x(t), t)

    Then:
        log p_0(x_0) ≈ log p_T(x_T) + ∫ div_x v(x(t), t) dt
    """
    device = x_0.device

    times = torch.linspace(eps, T, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x_t = torch.empty(
        (*x_0.shape, len(times)),
        dtype=x_0.dtype,
        device=device,
    )
    x_t[..., 0] = x_0

    log_det_correction = torch.zeros(x_0.shape[0], device=device)

    for n, t_scalar in enumerate(times[:-1]):
        t_batch = torch.ones(x_0.shape[0], device=device) * t_scalar

        x_current = x_t[..., n]

        def drift_fn(x_in, t_in):
            return ou_probability_flow_drift_for_likelihood(
                x_t=x_in,
                t=t_in,
                score_model=score_model,
                beta_schedule=beta_schedule,
            )

        drift = drift_fn(x_current, t_batch)
        divergence = exact_divergence(x_current, t_batch, drift_fn)

        x_t[..., n + 1] = x_current + drift * dt
        log_det_correction += divergence * dt

    sigma_T = ou_sigma_t(
        torch.ones(x_0.shape[0], device=device) * T,
        beta_integral=beta_integral,
    )
    log_p_T = gaussian_log_density(x_t[..., -1], sigma_T)

    log_p_0 = log_p_T + log_det_correction
    return log_p_0, times, x_t


# BPD computation

def loglik_to_bpd(
    log_p: torch.Tensor,
    image_shape: tuple[int, int, int] = (1, 28, 28),
    n_bits: int = 8,
) -> torch.Tensor:
    """
    Convert continuous log-likelihoods to bits per dimension (BPD)
    for images with discrete pixel values.

    For 8-bit images:
        BPD = -log p_cont(x) / (D log 2) + 8
    """
    D = np.prod(image_shape)
    return -log_p / (D * np.log(2.0)) + n_bits


def compute_bpd_for_model(
    config: ModelConfig,
    data_loader: DataLoader,
    device: str | torch.device,
    n_steps_likelihood: int = 100,
    eps: float = 1.0e-3,
):
    """
    Compute mean log-likelihood and mean BPD over a dataset for a model.
    """
    score_model = load_score_model(config, device=device)

    all_log_p = []
    all_bpd = []

    for x, _ in data_loader:
        x = x.to(device)

        if config.model_type == "bm":
            from bm_utils import compute_log_likelihood_ode

            log_p, _, _ = compute_log_likelihood_ode(
                x_0=x,
                score_model=score_model,
                sigma=config.sigma,
                T=1.0,
                n_steps=n_steps_likelihood,
                eps=eps,
            )

        elif config.model_type == "ou":
            beta_schedule, beta_integral = get_ou_schedule_functions(config.schedule_name)

            log_p, _, _ = compute_log_likelihood_ode_ou(
                x_0=x,
                score_model=score_model,
                beta_schedule=beta_schedule,
                beta_integral=beta_integral,
                T=1.0,
                n_steps=n_steps_likelihood,
                eps=eps,
            )
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

        bpd = loglik_to_bpd(log_p, image_shape=(1, 28, 28))

        all_log_p.append(log_p.detach().cpu())
        all_bpd.append(bpd.detach().cpu())

    all_log_p = torch.cat(all_log_p)
    all_bpd = torch.cat(all_bpd)

    return {
        "mean_log_likelihood": all_log_p.mean().item(),
        "std_log_likelihood": all_log_p.std().item(),
        "mean_bpd": all_bpd.mean().item(),
        "std_bpd": all_bpd.std().item(),
    }


def compare_model_configs_bpd(
    configs: list[ModelConfig],
    data_loader: DataLoader,
    device: str | torch.device,
    n_steps_likelihood: int = 100,
    eps: float = 1.0e-3,
) -> pd.DataFrame:
    """
    Compare multiple model configurations using BPD.
    """
    rows = []

    for config in configs:
        stats = compute_bpd_for_model(
            config=config,
            data_loader=data_loader,
            device=device,
            n_steps_likelihood=n_steps_likelihood,
            eps=eps,
        )

        rows.append({
            "model_type": config.model_type,
            "schedule_name": config.schedule_name,
            "checkpoint_path": config.checkpoint_path,
            **stats,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="mean_bpd", ascending=True).reset_index(drop=True)
    return df


# Sampling-comparison helper

def generate_final_samples(
    config: ModelConfig,
    sampler_name: str,
    n_images: int,
    device: str | torch.device,
    n_steps: int = 500,
    n_corrector_steps: int = 1,
    snr: float = 0.16,
):
    """
    Generate final samples for a given model and sampler.

    Returns:
        final_images: Tensor of shape (n_images, 1, 28, 28)
        synthetic_images_t: Full trajectories
    """
    score_model = load_score_model(config, device=device)

    if config.model_type == "bm":
        if sampler_name == "euler":
            _, synthetic_images_t = sample_bm_euler_maruyama(
                score_model=score_model,
                sigma=config.sigma,
                n_images=n_images,
                image_shape=(1, 28, 28),
                T=1.0,
                t_end=1.0e-3,
                n_steps=n_steps,
                device=device,
            )
        elif sampler_name == "predictor_corrector":
            _, synthetic_images_t = sample_bm_predictor_corrector(
                score_model=score_model,
                sigma=config.sigma,
                n_images=n_images,
                image_shape=(1, 28, 28),
                T=1.0,
                t_end=1.0e-3,
                n_steps=n_steps,
                n_corrector_steps=n_corrector_steps,
                snr=snr,
                device=device,
            )
        elif sampler_name == "ode":
            _, synthetic_images_t = sample_bm_probability_flow_ode(
                score_model=score_model,
                sigma=config.sigma,
                n_images=n_images,
                image_shape=(1, 28, 28),
                T=1.0,
                t_end=1.0e-3,
                n_steps=n_steps,
                device=device,
            )
        else:
            raise ValueError(f"Unknown sampler_name: {sampler_name}")

    elif config.model_type == "ou":
        beta_schedule, _ = get_ou_schedule_functions(config.schedule_name)

        if sampler_name == "euler":
            _, synthetic_images_t = sample_ou_euler_maruyama(
                score_model=score_model,
                beta_schedule=beta_schedule,
                n_images=n_images,
                image_shape=(1, 28, 28),
                T=1.0,
                t_end=1.0e-3,
                n_steps=n_steps,
                device=device,
            )
        elif sampler_name == "predictor_corrector":
            _, synthetic_images_t = sample_ou_predictor_corrector(
                score_model=score_model,
                beta_schedule=beta_schedule,
                n_images=n_images,
                image_shape=(1, 28, 28),
                T=1.0,
                t_end=1.0e-3,
                n_steps=n_steps,
                n_corrector_steps=n_corrector_steps,
                snr=snr,
                device=device,
            )
        elif sampler_name == "ode":
            _, synthetic_images_t = sample_ou_probability_flow_ode(
                score_model=score_model,
                beta_schedule=beta_schedule,
                n_images=n_images,
                image_shape=(1, 28, 28),
                T=1.0,
                t_end=1.0e-3,
                n_steps=n_steps,
                device=device,
            )
        else:
            raise ValueError(f"Unknown sampler_name: {sampler_name}")
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    final_images = synthetic_images_t[..., -1]
    return final_images, synthetic_images_t




# MNIST classifier for FID-MNIST and IS-MNIST

import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm


class MNISTClassifier(nn.Module):
    """
    Small CNN classifier for MNIST.

    It is used in Exercise 3 as:
    - a feature extractor for FID-MNIST,
    - a classifier for IS-MNIST.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def extract_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.view(x.shape[0], -1)
        features = F.relu(self.fc1(x))
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.fc2(features)
        return logits


def train_or_load_mnist_classifier(
    checkpoint_path: str,
    device: str | torch.device,
    n_epochs: int = 3,
    batch_size: int = 128,
):
    """
    Train or load a simple MNIST classifier.

    The classifier is not part of the generative model. It is only used
    to compute FID-MNIST and IS-MNIST.
    """
    classifier = MNISTClassifier().to(device)

    if os.path.exists(checkpoint_path):
        classifier.load_state_dict(torch.load(checkpoint_path, map_location=device))
        classifier.eval()
        print("Loaded MNIST classifier:", checkpoint_path)
        return classifier

    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=torch.get_num_threads(),
    )

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1.0e-3)
    criterion = nn.CrossEntropyLoss()

    classifier.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_correct = 0
        total_items = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = classifier(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_items += x.shape[0]

        avg_loss = total_loss / total_items
        acc = total_correct / total_items

        print(f"Epoch {epoch + 1}/{n_epochs} - loss={avg_loss:.4f} - acc={acc:.4f}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(classifier.state_dict(), checkpoint_path)

    classifier.eval()
    return classifier


def get_real_mnist_digit_images(
    digit: int,
    n_images: int,
    device: str | torch.device,
):
    """
    Get real MNIST images of a single digit.
    """
    data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    indices_digit = torch.where(data.targets == digit)[0][:n_images]
    images = torch.stack([data[i][0] for i in indices_digit]).to(device)

    return images


def compute_classifier_features_and_probs(
    classifier,
    images: torch.Tensor,
    batch_size: int = 128,
):
    """
    Compute classifier features and class probabilities.
    """
    classifier.eval()

    all_features = []
    all_probs = []

    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start:start + batch_size]

            features = classifier.extract_features(batch)
            logits = classifier(batch)
            probs = torch.softmax(logits, dim=1)

            all_features.append(features.cpu())
            all_probs.append(probs.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    probs = torch.cat(all_probs, dim=0).numpy()

    return features, probs


def compute_fid_from_features(real_features, generated_features):
    """
    Compute Fréchet distance between two Gaussian feature distributions.
    """
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)

    cov_real = np.cov(real_features, rowvar=False)
    cov_gen = np.cov(generated_features, rowvar=False)

    mean_diff = mu_real - mu_gen

    cov_sqrt = sqrtm(cov_real @ cov_gen)

    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = (
        mean_diff @ mean_diff
        + np.trace(cov_real + cov_gen - 2.0 * cov_sqrt)
    )

    return float(fid)


def compute_inception_score_from_probs(
    probs,
    eps: float = 1.0e-12,
):
    """
    Compute IS-MNIST from class probabilities.

    Note:
    For single-digit generation, IS should be interpreted carefully,
    because diversity across classes is not the objective.
    """
    p_y = np.mean(probs, axis=0, keepdims=True)

    kl = probs * (
        np.log(probs + eps) - np.log(p_y + eps)
    )

    score = np.exp(np.mean(np.sum(kl, axis=1)))

    return float(score)


def evaluate_fid_is_mnist(
    classifier,
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
):
    """
    Compute FID-MNIST and IS-MNIST for generated images.
    """
    real_features, _ = compute_classifier_features_and_probs(
        classifier,
        real_images,
    )

    generated_features, generated_probs = compute_classifier_features_and_probs(
        classifier,
        generated_images,
    )

    fid = compute_fid_from_features(real_features, generated_features)
    inception_score = compute_inception_score_from_probs(generated_probs)

    return {
        "fid_mnist": fid,
        "is_mnist": inception_score,
    }