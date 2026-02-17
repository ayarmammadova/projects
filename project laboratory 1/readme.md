# 🚦 Traffic Prediction with LSTM & GRU

Time-series forecasting of urban traffic conditions using recurrent neural networks.

---

## Overview

This project analyzes real traffic sensor data and predicts traffic speed and flow using deep learning models.

Two recurrent neural network architectures are compared:

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

The goal is to evaluate prediction accuracy and computational efficiency for intelligent transportation systems.

---

## Models

| Model | Description |
|------|------|
| LSTM | Captures long-term temporal dependencies |
| GRU | Simpler architecture with lower computational cost |

Result: GRU achieves comparable accuracy with lower complexity.

---

## Methodology

### Data
- Real traffic sensor measurements (speed and flow)
- One month period
- Time-series forecasting task

### Preprocessing
- Train / test split (80 / 20)
- Min-Max normalization
- Sliding window sequences

### Training
- Optimizer: Adam
- Input window: 20 time steps
- Regression prediction

---

## Evaluation

Metrics:
- RMSE
- MAPE

Both models capture temporal patterns with similar accuracy.

---

## Key Result

GRU provides similar prediction performance to LSTM while being computationally more efficient.

---

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

---

## My Work
- Data preprocessing
- Sequence generation
- Model training
- Evaluation and analysis
