# Generative AI Diffusion Models

This repository contains the implementation and evaluation of diffusion-based generative models for image generation on the MNIST dataset. The project focuses on score-based generative modeling using stochastic differential equations (SDEs) and their corresponding probability flow ordinary differential equations (ODEs).

## Project Overview

The objective of this project is to study and implement modern generative modeling techniques based on diffusion processes. The following components are included:

- Variance Exploding (VE) diffusion process based on Brownian motion
- Variance Preserving (VP) diffusion process based on the Ornstein-Uhlenbeck process
- Score-based neural network model for approximating the gradient of the log-density
- Sampling procedures using Euler-Maruyama integration and probability flow ODEs
- Quantitative evaluation using FID, Inception Score (IS), and Bits Per Dimension (BPD)

## Repository Structure
project_root/
├── notebooks/ # Notebooks containing experiments, results, and explanations
├── src/ # Source code (diffusion processes, models, utilities)
├── report/ # Final report in PDF format


## Installation and Requirements

The project requires the following Python libraries:

- torch
- torchvision
- numpy
- matplotlib

They can be installed using:

```bash
pip install torch torchvision numpy matplotlib

## Usage

Run the main notebook located in the `notebooks/` directory. The notebook is self-contained and includes the necessary steps for training, sampling, and evaluation.

## Notes

- The MNIST dataset is downloaded automatically during execution.
- Trained model checkpoints are not included in this repository due to size limitations.

## Results

The implemented models are capable of generating handwritten digit images. The quality of the generated samples is evaluated using FID, Inception Score (IS), and Bits Per Dimension (BPD).
