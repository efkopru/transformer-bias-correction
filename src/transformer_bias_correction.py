# %% [markdown]
# # Project: Transformer-based Temporal Bias Correction for S2S Forecasts
# 
# This project demonstrates a core requirement of the "AI in S2S Research Contractor" position:
# **Applying transformer-based bias correction to dynamical S2S model output.**
# 
# We build a simple Transformer model in PyTorch and train it to map biased synthetic raw model outputs to ground truth values.

# %% [code]
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset

# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## 1. Synthetic Data Generation
# We simulate a "forecast" from a dynamical model and a corresponding "truth" (like ERA5).
# Let's say the true temperature follows a sine wave + some noise. The dynamical forecast
# has a persistent bias (e.g., amplitude is off, and a phase shift or constant offset).

# %% [code]
days = 365 * 10 # 10 years of daily data
time = np.arange(days)
seasonality = np.sin(2 * np.pi * time / 365)

# Truth: seasonality + some random walk (weather noise)
truth = seasonality * 10 + np.random.randn(days).cumsum() * 0.1 + 15

# Forecast: Underestimates amplitude, has a positive bias, and smooths out noise
forecast = seasonality * 7 + 18 + np.random.randn(days) * 0.5

# Let's visualize a small slice
plt.figure(figsize=(12, 5))
plt.plot(time[:365], truth[:365], label="Truth (ERA5)", alpha=0.8)
plt.plot(time[:365], forecast[:365], label="Raw Forecast (CFSv2/NASA mock)", alpha=0.8)
plt.title("Synthetic S2S Temperature Forecast vs Truth (1 Year slice)")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.savefig("assets/data_sample.png")
plt.show()

# %% [markdown]
# ## 2. Data Preparation for PyTorch
# We'll frame this as a sequence-to-sequence task. We take windows of 30 days of the forecast and try to predict the same window of truth.

# %% [code]
SEQ_LEN = 30 # 30 days of data at a time

def create_sequences(forecast_data, truth_data, seq_len):
    X = []
    y = []
    for i in range(len(forecast_data) - seq_len):
        X.append(forecast_data[i:i+seq_len])
        y.append(truth_data[i:i+seq_len])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

X, y = create_sequences(forecast, truth, SEQ_LEN)

# Train/Test Split (80/20)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %% [markdown]
# ## 3. Transformer Model Architecture
# A lightweight Transformer designed for sequence-to-sequence regression.

# %% [code]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class S2SBiasCorrectionTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, input_dim)

    def forward(self, src):
        # src shape: (batch, seq_len, 1)
        x = self.input_linear(src)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = self.output_linear(x)
        return out

model = S2SBiasCorrectionTransformer()
print("Model initialized.")

# %% [markdown]
# ## 4. Training Loop
# We train the model using Mean Squared Error (MSE) loss.

# %% [code]
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    batch_losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    avg_loss = np.mean(batch_losses)
    train_losses.append(avg_loss)
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.savefig("assets/loss.png")
plt.show()

# %% [markdown]
# ## 5. Evaluation and Results
# Let's apply the trained transformer to the test set to evaluate its bias correction capability.

# %% [code]
model.eval()
test_preds = []
test_actuals = []
test_raw = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        preds = model(batch_X)
        test_preds.append(preds.squeeze().numpy())
        test_actuals.append(batch_y.squeeze().numpy())
        test_raw.append(batch_X.squeeze().numpy())

# Flatten
test_preds = np.concatenate(test_preds).flatten()
test_actuals = np.concatenate(test_actuals).flatten()
test_raw = np.concatenate(test_raw).flatten()

# Calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

rmse_raw = rmse(test_actuals, test_raw)
rmse_corrected = rmse(test_actuals, test_preds)

print(f"RMSE of Raw Forecast: {rmse_raw:.4f}")
print(f"RMSE of Transformer Corrected Forecast: {rmse_corrected:.4f}")

# %% [markdown]
# ### Visualizing the Correction

# %% [code]
plt.figure(figsize=(14, 6))
# Plot just the first 100 days of test set for clarity
subset = 100
plt.plot(test_actuals[:subset], label='Truth (ERA5)', color='black', linewidth=2)
plt.plot(test_raw[:subset], label=f'Raw Forecast (RMSE={rmse_raw:.2f})', color='red', linestyle='--')
plt.plot(test_preds[:subset], label=f'Transformer Corrected (RMSE={rmse_corrected:.2f})', color='green', alpha=0.8)

plt.title("Bias Correction Results on Unseen Data")
plt.xlabel("Days")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.savefig("assets/results.png")
plt.show()

# %% [markdown]
# ### Additional Error Analysis
# To better understand the transformer's performance, let's look at the error distributions and a scatter plot of predictions versus actuals.

# %% [code]
# Calculate errors
error_raw = test_raw - test_actuals
error_corrected = test_preds - test_actuals

plt.figure(figsize=(10, 5))
sns.kdeplot(error_raw, label="Raw Error", fill=True, color="red", alpha=0.3)
sns.kdeplot(error_corrected, label="Corrected Error", fill=True, color="green", alpha=0.4)
plt.axvline(0, color='black', linestyle='--')
plt.title("Error Distribution (Prediction - Truth)")
plt.xlabel("Error (°C)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig("assets/error_dist.png")
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(test_actuals, test_raw, alpha=0.1, label="Raw Forecast", color="red")
plt.scatter(test_actuals, test_preds, alpha=0.1, label="Corrected Forecast", color="green")
# 1:1 line
min_val = min(test_actuals.min(), test_raw.min(), test_preds.min())
max_val = max(test_actuals.max(), test_raw.max(), test_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label="Ideal (1:1)")
plt.title("Scatter Plot: Actual vs Forecast")
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Forecasted Temperature (°C)")
plt.legend()
plt.grid(True)
plt.savefig("assets/scatter.png")
plt.show()

# %% [markdown]
# ## Conclusion
# By adapting sequence-to-sequence Transformer architectures, we can successfully map biased dynamical outputs to align with ground truth datasets. In a real-world scenario, this would apply to higher-dimensional NetCDF grids using 3D Attention or Vision Transformers (ViTs).
