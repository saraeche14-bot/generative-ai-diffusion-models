# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 17:26:43 2025

@author: ALBERTO
"""

from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib.colors import Colormap

import torch
from  torchvision.utils import make_grid
from torchvision.transforms import functional

def plot_image_grid( 
    images: torch.Tensor, 
    figsize: tuple,
    n_rows: int,
    n_cols: int,
    padding: int = 2,
    pad_value: int = 1.0,
    cmap: Colormap = "gray",
    normalize: bool = False,
    axis_on_off: bool = "off",  
 ):

    grid = make_grid(
        images, 
        nrow=n_cols, 
        padding=padding, 
        normalize=normalize,
        pad_value=pad_value,
    ) 

    # Convert to PIL Image and display

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(functional.to_pil_image(grid), cmap=cmap)
    ax.axis("off")
    return fig, ax


def plot_image_evolution(
    images: torch.Tensor,
    n_images: int,
    n_intermediate_steps: ArrayLike,
    figsize: tuple,
    cmap: Colormap = "gray",
):
    fig, axs = plt.subplots(
        n_images, 
        len(n_intermediate_steps), 
        figsize=figsize,
    )

    for n_image in np.arange(n_images):
        for i, ax in enumerate(axs[n_image, :]):
            ax.imshow(
                images[n_image, 0,:, :, n_intermediate_steps[i]], 
                cmap="gray",
                )
            axs[n_image, i].set_axis_off()
    return fig, axs
    
def animation_images(
        images_t, 
        interval,
        figsize,
    ): 
    _, _, n_frames = np.shape(images_t)

    # Create a figure and axes.  
    fig, ax = plt.subplots(figsize=figsize)
    img_display = ax.imshow(images_t[:, :, 0], cmap="gray")

    def update(t):
        """Update function for the animation."""
        img_display.set_array(images_t[:, :, t])
        return [img_display]

    return ( 
        fig, 
        ax, 
        animation.FuncAnimation(
            fig, 
            update, 
            frames=n_frames, 
            interval=interval, 
            blit=False)
    )
