import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, BatchNormalization, LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import timedelta

# Configure logging
logging.basicConfig(filename="stock_prediction.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Cache for stock data
cache = {}

def fetch_stock_data_cached(ticker, period="1y"):
    if ticker in cache:
        return cache[ticker]
    
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    if data.shape[0] < 100:
        return {"error": f"Not enough historical data for {ticker} ({data.shape[0]} records)."}
    
    # Handling missing values
    data = data.interpolate(method='linear')
    data.dropna(inplace=True)
    
    # Technical indicators
    data["SMA_20"] = data["Close"].rolling(20, min_periods=1).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["Momentum"] = data["Close"].diff(5)
    
    # RSI Indicator
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    data["RSI"] = 100 - (100 / (1 + rs))
    
    data.dropna(inplace=True)
    if data.shape[0] < 100:
        return {"error": f"Not enough processed data for {ticker}, only {data.shape[0]} rows after cleaning."}
    
    cache[ticker] = data
    return data

def preprocess_data(data):
    features = ["Close", "SMA_20", "EMA_20", "Momentum", "RSI"]  # Removed 'Volume'
    
    for f in features:
        if f not in data.columns:
            data[f] = 0  # Fill missing features with zeros
    
    print("✅ Available Columns in Data:", list(data.columns))
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    close_scaler = MinMaxScaler()
    close_scaler.fit(data[["Close"]])
    return scaled_data, scaler, close_scaler

def create_sequences(scaled_data, seq_length=50):
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length][0])
    return np.array(X), np.array(y)

def build_model(seq_length, feature_size):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(seq_length, feature_size)),
        BatchNormalization(),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        GRU(64, return_sequences=False),  # Removed incorrect input_shape
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_or_load_model(X_train, y_train, seq_length=50, model_path="models/stock_model.keras"):
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        logging.info("Loaded pre-trained model.")
        return model
    
    model = build_model(seq_length, X_train.shape[2])
    print("✅ Model Input Shape:", X_train.shape)
    
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    model.save(model_path)
    return model

def evaluate_model(model, X_test, y_test, close_scaler):
    y_pred_scaled = model.predict(X_test)
    y_pred_inv = close_scaler.inverse_transform(y_pred_scaled)
    y_test_inv = close_scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    smape = np.mean(2 * np.abs(y_test_inv - y_pred_inv) / (np.abs(y_test_inv) + np.abs(y_pred_inv) + 1e-10)) * 100
    
    return {
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "SMAPE": round(float(smape), 2)
    }

def backtest_prediction(ticker):
    data_df = fetch_stock_data_cached(ticker, period="6mo")
    if isinstance(data_df, dict) and "error" in data_df:
        return data_df

    scaled_data, _, close_scaler = preprocess_data(data_df)
    seq_length = 50
    X, y = create_sequences(scaled_data, seq_length)
    
    if len(X) == 0:
        return {"error": "Not enough data after creating sequences."}
    
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    model_instance = train_or_load_model(X_train, y_train)
    metrics = evaluate_model(model_instance, X_test, y_test, close_scaler)
    
    return {"ticker": ticker, "accuracy": metrics}

def predict_future_stream(ticker, future_days=30, model_path="models/stock_model.keras"):
    data_df = fetch_stock_data_cached(ticker, period="1y")
    if isinstance(data_df, dict) and "error" in data_df:
        return [data_df]

    scaled_data, _, close_scaler = preprocess_data(data_df)
    seq_length = 50
    X, y = create_sequences(scaled_data, seq_length=seq_length)

    if len(X) == 0:
        return [{"error": "Not enough data after creating sequences."}]

    model_instance = train_or_load_model(X, y, seq_length, model_path)
    last_sequence = scaled_data[-seq_length:]
    last_date = data_df.index[-1].date()
    predictions = []

    for _ in range(future_days):
        last_date += timedelta(days=1)
        if last_date.weekday() >= 5:
            continue

        pred_scaled = model_instance.predict(last_sequence.reshape(1, seq_length, scaled_data.shape[1]))
        pred_price = close_scaler.inverse_transform(pred_scaled)[0, 0]
        predictions.append({"date": str(last_date), "prediction": round(float(pred_price), 2)})
        
        new_row = last_sequence[-1]
        new_row[0] = pred_scaled[0, 0]
        last_sequence = np.vstack([last_sequence[1:], new_row])
    
    return predictions
