import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import yfinance as yf
from datetime import timedelta

# Define LSTM Model
class LSTMStockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(LSTMStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])

# Fetch historical stock data
def fetch_stock_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.shape[0] < 200:
        return {"error": f"Not enough historical data for {ticker}"}

    df["SMA_20"] = df["Close"].rolling(20, min_periods=5).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Momentum"] = df["Close"].diff(5).fillna(0)
    df.dropna(inplace=True)
    return df

# Preprocess Data
def preprocess_data(df):
    features = ["Open", "Close", "SMA_20", "EMA_20", "Momentum"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    return scaled_data, scaler

# Create Sequences for LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][1])  # Predicting Close price
    return np.array(X), np.array(y)

# Train LSTM Model
def train_model(ticker):
    df = fetch_stock_data(ticker, period="2y")
    if isinstance(df, dict) and "error" in df:
        return df

    scaled_data, scaler = preprocess_data(df)
    X, y_actual = create_sequences(scaled_data)

    X_train, y_train = X[:-30], y_actual[:-30]  # Train on all but last 30 days
    X_test, y_test_actual = X[-30:], y_actual[-30:]

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = LSTMStockPredictor(input_dim=X.shape[2])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):  # Train for 50 epochs
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, scaler
def backtest_model(ticker):
    df = fetch_stock_data(ticker, period="2y")
    if isinstance(df, dict) and "error" in df:
        return df

    scaled_data, scaler = preprocess_data(df)
    X, y_actual = create_sequences(scaled_data)

    X_test, y_test_actual = X[-60:], y_actual[-60:]  # Last 60 days for backtest

    model, scaler = train_model(ticker)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_pred_scaled = model(X_test_tensor).detach().numpy().flatten()
    y_test_actual = np.array(y_test_actual).flatten()

    # Ensure correct shape transformation
    dummy_features = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
    dummy_features[:, 1] = y_pred_scaled
    y_pred_actual = scaler.inverse_transform(dummy_features)[:, 1]

    dummy_features[:, 1] = y_test_actual
    y_test_actual = scaler.inverse_transform(dummy_features)[:, 1]

    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)

    return {
        "ticker": ticker,
        "test_dates": list(df.index[-len(y_test_actual):].strftime('%Y-%m-%d')),
        "actual_close_prices": y_test_actual.tolist(),
        "predicted_close_prices": y_pred_actual.tolist(),
        "accuracy": {"MSE": round(mse, 4), "RMSE": round(rmse, 4)}
    }
def predict_future(ticker, days=30):
    df = fetch_stock_data(ticker, period="2y")
    if isinstance(df, dict) and "error" in df:
        return df

    scaled_data, scaler = preprocess_data(df)
    X, _ = create_sequences(scaled_data)
    last_sequence = X[-1]  # Last sequence for prediction

    model, scaler = train_model(ticker)

    predictions = []
    last_date = df.index[-1].date()

    for _ in range(days):
        last_date += timedelta(days=1)
        if last_date.weekday() >= 5:  # Skip weekends
            continue

        X_tensor = torch.tensor(last_sequence.reshape(1, *last_sequence.shape), dtype=torch.float32)
        pred_scaled = model(X_tensor).detach().numpy().flatten()

        # Correct scaling conversion
        dummy_features = np.zeros((1, scaled_data.shape[1]))
        dummy_features[:, 1] = pred_scaled
        pred_actual = scaler.inverse_transform(dummy_features)[:, 1]

        predictions.append({"date": str(last_date), "close_prediction": round(pred_actual[0], 2)})

        # Update sequence with the new prediction
        new_row = last_sequence[-1].copy()
        new_row[1] = pred_scaled[0]
        last_sequence = np.vstack([last_sequence[1:], new_row])

    return predictions
