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

logging.basicConfig(filename="stock_prediction.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

cache = {}

def fetch_stock_data_cached(ticker, period="5y"):
    if ticker in cache:
        return cache[ticker]

    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    if data.shape[0] < 200:
        return {"error": f"Not enough historical data for {ticker} ({data.shape[0]} records)."}

    data = data.interpolate(method='linear')
    data.dropna(inplace=True)

    # Technical Indicators
    data["SMA_20"] = data["Close"].rolling(20, min_periods=5).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["Momentum"] = data["Close"].diff(5).fillna(0)

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=5).mean()
    avg_loss = loss.rolling(14, min_periods=5).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    data["RSI"] = 100 - (100 / (1 + rs))

    data.dropna(inplace=True)

    if data.shape[0] < 100:
        return {"error": f"Not enough processed data for {ticker}, only {data.shape[0]} rows after cleaning."}

    cache[ticker] = data
    return data

def preprocess_data(data):
    features = ["Open", "Close", "SMA_20", "EMA_20", "Momentum", "RSI"]
    
    for f in features:
        if f not in data.columns:
            data[f] = 0

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    open_scaler = MinMaxScaler()
    close_scaler = MinMaxScaler()
    open_scaler.fit(data[["Open"]])
    close_scaler.fit(data[["Close"]])

    return scaled_data, scaler, open_scaler, close_scaler

def create_sequences(scaled_data, seq_length=60):
    X, y_open, y_close = [], [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y_open.append(scaled_data[i+seq_length][0])  # Open price
        y_close.append(scaled_data[i+seq_length][1])  # Close price
    return np.array(X), np.array(y_open), np.array(y_close)

def build_model(seq_length, feature_size):
    model = Sequential([
        Conv1D(filters=128, kernel_size=3, activation="relu", input_shape=(seq_length, feature_size)),
        BatchNormalization(),
        LSTM(256, return_sequences=True),
        BatchNormalization(),
        GRU(128, return_sequences=False),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(2)  # Predicting both Open & Close
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_or_load_model(X_train, y_open_train, y_close_train, seq_length=60, model_path="models/stock_model_v2.keras"):
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        logging.info("Loaded pre-trained model.")
        return model
    
    model = build_model(seq_length, X_train.shape[2])

    model.fit(X_train, np.column_stack((y_open_train, y_close_train)), epochs=150, batch_size=32, verbose=1)
    model.save(model_path)
    return model

def predict_future_stream(ticker, future_days=30, model_path="models/stock_model_v2.keras"):
    data_df = fetch_stock_data_cached(ticker, period="2y")
    if isinstance(data_df, dict) and "error" in data_df:
        return [data_df]

    scaled_data, _, open_scaler, close_scaler = preprocess_data(data_df)
    seq_length = 60
    X, y_open, y_close = create_sequences(scaled_data, seq_length=seq_length)

    if len(X) == 0:
        return [{"error": "Not enough data after creating sequences."}]

    model_instance = train_or_load_model(X, y_open, y_close, seq_length, model_path)
    last_sequence = scaled_data[-seq_length:]
    last_date = data_df.index[-1].date()
    predictions = []

    days_generated = 0
    while days_generated < future_days:
        last_date += timedelta(days=1)
        if last_date.weekday() >= 5:
            continue

        pred_scaled = model_instance.predict(last_sequence.reshape(1, seq_length, scaled_data.shape[1]))
        pred_open = open_scaler.inverse_transform([[pred_scaled[0][0]]])[0][0]
        pred_close = close_scaler.inverse_transform([[pred_scaled[0][1]]])[0][0]

        predictions.append({
            "date": str(last_date),
            "open_prediction": round(float(pred_open), 2),
            "close_prediction": round(float(pred_close), 2)
        })
        days_generated += 1

        new_row = last_sequence[-1]
        new_row[0] = pred_scaled[0][0]
        new_row[1] = pred_scaled[0][1]
        last_sequence = np.vstack([last_sequence[1:], new_row])

    return predictions
