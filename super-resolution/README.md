  Super-Resolution Project

Deep learning pipeline for Single Image Super-Resolution (SISR) with dataset preparation, baseline evaluation and comparison of modern SR architectures.

⸻
## Overview

This project implements an end-to-end super-resolution workflow:

1. Dataset preparation (CelebA → LR/HR pairs)
2. Baseline reconstruction (Bicubic)
3. Deep learning training (SRGAN / ESRGAN / EDSR)
4. Quantitative and visual evaluation

The goal is to compare perceptual and reconstruction quality across different model families (CNN vs GAN).

⸻

## Models

| Model | Purpose |
|------|------|
| Bicubic | Baseline reference |
| SRGAN | First GAN-based SR model |
| ESRGAN | Perceptual quality improvement |
| **EDSR** | Final selected model (highest PSNR, stable training) |

Final choice: EDSR provides the best reconstruction reliability while avoiding GAN artifacts.

⸻

## Evaluation

**Metrics used:**

- PSNR
- SSIM
- Visual comparison

The project compares reconstruction accuracy vs perceptual sharpness trade-offs.

⸻

## My Contribution (Team Project)

- ESRGAN experiments and tuning
- PSNR / SSIM evaluation pipeline
- Results analysis
- Report writing

⸻

## Project Structure

SuperResolution_Project/
├── notebooks/          # Training & preprocessing pipelines
├── src/                # Dataset loaders and model implementations
├── experiments/        # Logs and configs
├── report/             # Final comparison results
├── requirements.txt
└── README.md


⸻
## Tech Stack

- Python, PyTorch, torchvision
- NumPy, matplotlib, Pillow
- scikit-image
- Jupyter Notebook

⸻

 Key Result

EDSR achieved the best reconstruction quality, outperforming GAN-based models in stability and PSNR while avoiding perceptual artifacts.
