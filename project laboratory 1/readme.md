⸻

🚦 Traffic Prediction with LSTM & GRU

Time-series forecasting of urban traffic conditions using recurrent neural networks.

⸻

📖 Overview

This project analyzes real traffic sensor data and predicts traffic speed and flow using deep learning time-series models.

The task is formulated as a supervised forecasting problem where future traffic values are predicted from historical measurements.
Two recurrent neural network architectures are compared:
	•	LSTM (Long Short-Term Memory)
	•	GRU (Gated Recurrent Unit)

The goal is to evaluate prediction accuracy and computational efficiency of both models in intelligent transportation systems.

⸻

🧠 Models

Model	Description
LSTM	Captures long-term temporal dependencies using memory cells
GRU	Simpler gated architecture with lower computational cost

Result: GRU achieves comparable accuracy with lower complexity.

⸻

⚙️ Methodology

Data
	•	Real traffic sensor measurements (speed & flow)
	•	One month time period
	•	Time-series forecasting setup

Preprocessing
	•	Train / test split (80 / 20)
	•	Min-Max normalization
	•	Sliding window sequence generation

Training
	•	Optimizer: Adam
	•	Input window: 20 time steps
	•	Regression prediction (next value forecasting)

⸻

📊 Evaluation

Metrics:
	•	RMSE (Root Mean Squared Error)
	•	MAPE (Mean Absolute Percentage Error)



⸻

📈 Key Result

GRU provided similar prediction performance to LSTM while being computationally more efficient, making it more suitable for real-time traffic forecasting systems.  ￼

⸻

🛠️ Tech Stack
	•	Python
	•	TensorFlow / Keras
	•	NumPy, Pandas
	•	Matplotlib
	•	scikit-learn

⸻

👤 My Work
	•	Data preprocessing & normalization
	•	Time-series window generation
	•	Model training and comparison
	•	Evaluation and analysis


