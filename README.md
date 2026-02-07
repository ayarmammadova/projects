
# SuperResolution_Project

**Milestone 1, 2 & Final Project – Dataset Preparation, Baseline & Deep Learning Super-Resolution**

---

##  Project Description

This project implements a complete **Single Image Super-Resolution (SISR)** pipeline using deep learning.
The work progresses through **Milestone 1 and Milestone 2**, and is further extended into a **final comparative study of advanced super-resolution models**.

The project uses **CelebA** (primary dataset) and **DIV2K** (optional) and focuses on both **quantitative evaluation (PSNR, SSIM)** and **qualitative visual comparison**.

---

##  Milestone 1 — Dataset Acquisition & Preparation

* Download **CelebA** dataset using Kaggle API
* Extract and align face images
* Generate **Low-Resolution (LR)** and **High-Resolution (HR)** image pairs (2× super-resolution)
* Split dataset into **train / validation / test** sets
* Visualize LR vs HR image pairs

---

##  Milestone 2 — Modeling Pipeline

* Unified **PyTorch Dataset & DataLoader**
* Baseline super-resolution using **Bicubic Interpolation**
* Deep learning model training using **SRGAN**
* Evaluation using **PSNR** and **SSIM**
* End-to-end training and evaluation notebooks
* Clear and reproducible pipeline

---

##  Final Extension — Model Comparison & Advanced Training

After completing Milestone 2, the project was **extended** to include a **comparative study of modern super-resolution models**:

### Models Implemented

* **EDSR (Enhanced Deep Super-Resolution Network)**

  * Residual-based CNN optimized for pixel-wise accuracy
  * High PSNR and stable training
  * Selected as the **main model** of the project

* **ESRGAN (Enhanced SRGAN)**

  * GAN-based model optimized for perceptual quality
  * Produces sharper textures and higher SSIM
  * Included as an **extended experimental model**

---

## Model Selection Rationale

Although **ESRGAN** achieves competitive **SSIM** values and sharper perceptual results, it requires longer training time and may introduce artifacts.

**EDSR** demonstrates:

* Higher PSNR
* More stable convergence
* More reliable reconstruction quality

 **Final Decision:**

* **EDSR is selected as the primary model**
* **ESRGAN is presented as an extended comparison experiment**

This decision aligns with common practices in super-resolution research.

---

##  Team Information

**Team Name:** PixelForge

## Team Project

This project was completed as part of a university team.

**My contribution:**
- ESRGAN experiments
- Evaluation using PSNR / SSIM
- Results analysis
- Report writing
  

##  Repository Structure

```
SuperResolution_Project/
│
├── data/                         # (ignored by git) raw & processed datasets
│
├── notebooks/
│   ├── 01_celeba_prep.ipynb      # Milestone 1: CelebA preprocessing
│   ├── 02_div2k_prep.ipynb       # Optional: DIV2K preprocessing
│   ├── 03_srgan_training.ipynb   # SRGAN / ESRGAN training
│   └── 04_baseline_bicubic.ipynb # Bicubic baseline
│
├── experiments/                  # Saved experiment configs & logs
│
├── report/                       # PDF reports & comparison results
│
├── src/
│   ├── dataset.py                # Unified PyTorch dataset
│   ├── CelebASRDataset.py        # CelebA loader
│   ├── DIV2KSRDataset.py         # DIV2K loader
│   ├── srgan_model.py            # SRGAN / ESRGAN models
│   └── transforms.py             # Image transforms
│
├── .gitignore
├── requirements.txt
├── kaggle.json                   # (ignored) Kaggle API key
└── README.md
```

---

##  Software Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Main Libraries

* Python 3.10+
* PyTorch / torchvision
* numpy
* matplotlib
* Pillow
* tqdm
* scikit-image
* Kaggle API
* Jupyter Notebook

---

##  How to Run the Pipeline

### 1️⃣ Dataset Preparation (Milestone 1)

Run:

```
notebooks/01_celeba_prep.ipynb
```

This will:

* Download CelebA
* Align and crop faces
* Generate LR/HR pairs
* Create train / val / test splits

---

### 2️⃣ Baseline Model (Milestone 2)

Run:

```
notebooks/04_baseline_bicubic.ipynb
```

This notebook:

* Upscales LR images using bicubic interpolation
* Computes PSNR / SSIM
* Saves visual comparison samples

---

### 3️⃣ Deep Learning Models

#### SRGAN / ESRGAN

Run:

```
notebooks/03_srgan_training.ipynb
```

This notebook:

* Trains SRGAN / ESRGAN models
* Logs generator and discriminator losses
* Evaluates PSNR and SSIM
* Produces SR visualizations

#### EDSR

EDSR is trained using the same dataset and evaluation pipeline and serves as the **final selected model**.

---

##  Results Summary

### Baseline (Bicubic Interpolation)

* Fast and simple
* Limited detail reconstruction
* Used as reference

### SRGAN / ESRGAN

* Improved perceptual sharpness
* Higher SSIM
* Longer training time

### EDSR (Final Model)

* Highest PSNR
* Stable and clean reconstructions
* Best overall trade-off between quality and robustness

Detailed numerical results and visual comparisons are included in the **report/** directory.

---

##  Notes on Dataset & Storage

Due to size constraints:

* `/data`
* `/outputs`
* model checkpoints
* image folders

are excluded from Git tracking.

Only **code, notebooks, configuration files, and documentation** are version-controlled.

---

##  Use of LLMs

Large Language Models (e.g., ChatGPT) were used for:

* Code commenting assistance
* Documentation polishing
* English language refinement

All outputs were manually reviewed and validated.

---

##  Conclusion

This project demonstrates a full super-resolution workflow from dataset preparation to advanced model comparison.

By selecting **EDSR as the primary model** and including **SRGAN / ESRGAN as extended experiments**, the project balances:

* theoretical correctness
* practical implementation
* and reproducibility

