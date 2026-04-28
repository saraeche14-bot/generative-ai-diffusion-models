# -*- coding: utf-8 -*-
"""
Colored MNIST dataset for conditional diffusion models.

This module builds an RGB version of MNIST by assigning a color to each
black-and-white digit image. The digit label and the color label are both
returned, so the diffusion model can be conditioned on digit and color.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


COLOR_PALETTE = {
    "red": torch.tensor([1.0, 0.0, 0.0]),
    "green": torch.tensor([0.0, 1.0, 0.0]),
    "blue": torch.tensor([0.0, 0.0, 1.0]),
    "yellow": torch.tensor([1.0, 1.0, 0.0]),
    "cyan": torch.tensor([0.0, 1.0, 1.0]),
    "magenta": torch.tensor([1.0, 0.0, 1.0]),
    "white": torch.tensor([1.0, 1.0, 1.0]),
}


COLOR_NAMES = list(COLOR_PALETTE.keys())
COLOR_TO_ID = {name: idx for idx, name in enumerate(COLOR_NAMES)}
ID_TO_COLOR = {idx: name for name, idx in COLOR_TO_ID.items()}


class ColoredMNIST(Dataset):
    """
    RGB version of MNIST.

    Each original grayscale MNIST image x with shape (1, 28, 28) is converted
    into a colored RGB image with shape (3, 28, 28). The digit strokes are
    multiplied by an RGB color vector.

    The dataset returns:
        colored_image: Tensor of shape (3, 28, 28)
        digit_label: integer in {0, ..., 9}
        color_label: integer in {0, ..., num_colors - 1}
    """

    def __init__(
        self,
        mnist_dataset: Dataset,
        color_mode: str = "random",
        fixed_color: str | None = None,
        seed: int = 123,
    ):
        """
        Args:
            mnist_dataset:
                Original torchvision MNIST dataset with ToTensor transform.
            color_mode:
                "random": each image receives a random color from the palette.
                "by_digit": each digit is always assigned the same color.
                "fixed": all images receive the same color.
            fixed_color:
                Color used when color_mode="fixed".
            seed:
                Seed for reproducible random color assignment.
        """
        if color_mode not in {"random", "by_digit", "fixed"}:
            raise ValueError("color_mode must be 'random', 'by_digit' or 'fixed'.")

        if color_mode == "fixed" and fixed_color not in COLOR_TO_ID:
            raise ValueError(f"fixed_color must be one of {COLOR_NAMES}.")

        self.mnist_dataset = mnist_dataset
        self.color_mode = color_mode
        self.fixed_color = fixed_color
        self.num_colors = len(COLOR_NAMES)

        generator = torch.Generator()
        generator.manual_seed(seed)

        if color_mode == "random":
            self.color_labels = torch.randint(
                low=0,
                high=self.num_colors,
                size=(len(mnist_dataset),),
                generator=generator,
            )
        elif color_mode == "by_digit":
            self.color_labels = None
        else:
            fixed_color_id = COLOR_TO_ID[fixed_color]
            self.color_labels = torch.full(
                size=(len(mnist_dataset),),
                fill_value=fixed_color_id,
                dtype=torch.long,
            )

    def __len__(self) -> int:
        return len(self.mnist_dataset)

    def __getitem__(self, index: int):
        image_gray, digit_label = self.mnist_dataset[index]

        if image_gray.shape[0] != 1:
            raise ValueError("Expected MNIST images with shape (1, 28, 28).")

        digit_label = int(digit_label)

        if self.color_mode == "random":
            color_label = int(self.color_labels[index])
        elif self.color_mode == "by_digit":
            color_label = digit_label % self.num_colors
        else:
            color_label = int(self.color_labels[index])

        color_name = ID_TO_COLOR[color_label]
        color_rgb = COLOR_PALETTE[color_name].to(
            device=image_gray.device,
            dtype=image_gray.dtype,
        )

        image_rgb = image_gray.repeat(3, 1, 1)
        image_colored = image_rgb * color_rgb[:, None, None]

        return (
            image_colored,
            torch.tensor(digit_label, dtype=torch.long),
            torch.tensor(color_label, dtype=torch.long),
        )


def get_color_names() -> list[str]:
    """Return the available color names."""
    return COLOR_NAMES


def get_color_to_id() -> dict[str, int]:
    """Return the mapping from color names to integer labels."""
    return COLOR_TO_ID


def get_id_to_color() -> dict[int, str]:
    """Return the mapping from integer labels to color names."""
    return ID_TO_COLOR