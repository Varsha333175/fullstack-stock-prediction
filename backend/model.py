import os
import time
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from datetime import timedelta

def fetch_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    if data.shape[0] < 30:
        return {"error": f"Not enough historical data for {ticker} ({data.shape[0]} records)."}
    
    # Basic technical indicators
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["RSI"] = 100 - (100 / (1 + (data["Close"].diff().clip(lower=0).rolling(14).mean() /
                                     data["Close"].diff().clip(upper=0).abs().rolling(14).mean())))
    
    # Bollinger Bands (20-day)
    data["BB_mid"] = data["Close"].rolling(20).mean()
    data["BB_std"] = data["Close"].rolling(20).std()
    data["BB_up"] = data["BB_mid"] + 2 * data["BB_std"]
    data["BB_down"] = data["BB_mid"] - 2 * data["BB_std"]
    
    # MACD
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD_line"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_line"] = data["MACD_line"].ewm(span=9, adjust=False).mean()
    
    # Stochastic Oscillator
    data["High_14"] = data["High"].rolling(14).max()
    data["Low_14"] = data["Low"].rolling(14).min()
    data["%K"] = (data["Close"] - data["Low_14"]) * 100 / (data["High_14"] - data["Low_14"])
    data["%D"] = data["%K"].rolling(3).mean()
    
    data.dropna(inplace=True)
    if data.shape[0] < 20:
        return {"error": f"Not enough processed data for {ticker}, only {data.shape[0]} rows after cleaning."}
    return data

def preprocess_data(data):
    features = [
        "Close", "SMA_10", "EMA_10", "RSI",
        "BB_up", "BB_down",
        "MACD_line", "Signal_line",
        "%K", "%D",
        "Volume"
    ]
    # Ensure all expected features exist
    for f in features:
        if f not in data.columns:
            data[f] = 0
    # Scaler for all features (used for creating sequences)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    # Dedicated scaler for the target "Close" price only.
    close_scaler = MinMaxScaler()
    close_scaler.fit(data[["Close"]])
    
    return scaled_data, scaler, close_scaler

def create_sequences(scaled_data, seq_length=10):
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length][0])  # Predict "Close" price (first feature)
    return np.array(X), np.array(y)

def train_or_load_model(X_train, y_train, model_path="saved_lstm_model.h5"):
    # If an old model exists with a different input shape, delete it.
    if os.path.exists(model_path):
        loaded_model = keras.models.load_model(model_path)
        expected_features = X_train.shape[2]
        if loaded_model.input_shape[-1] != expected_features:
            print("[INFO] Existing model input shape does not match expected features. Deleting model.")
            os.remove(model_path)
        else:
            print("[INFO] Loading existing LSTM model from disk...")
            loaded_model.compile(optimizer="adam", loss="mean_squared_error")
            return loaded_model
    print("[INFO] Training a new LSTM model with advanced indicators and adjusted dropout...")
    # Reduced dropout rate to 0.2 to avoid underfitting
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)
    model.save(model_path)
    return model

def evaluate_model(model, X_test, y_test, close_scaler):
    y_pred_scaled = model.predict(X_test)
    # Inverse transform using the dedicated close scaler
    y_pred_inv = close_scaler.inverse_transform(y_pred_scaled)
    y_test_scaled = np.array(y_test).reshape(-1, 1)
    y_test_inv = close_scaler.inverse_transform(y_test_scaled)
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    return {
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
        "R2": round(float(r2), 4)
    }, y_test_inv, y_pred_inv

def backtest_prediction(ticker):
    data_df = fetch_stock_data(ticker, period="10y")
    if isinstance(data_df, dict) and "error" in data_df:
        return data_df
    scaled_data, scaler, close_scaler = preprocess_data(data_df)
    seq_length = 10
    X, y = create_sequences(scaled_data, seq_length)
    if len(X) == 0:
        return {"error": "Not enough data after creating sequences."}
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    model_instance = train_or_load_model(X_train, y_train)
    metrics, y_test_inv, y_pred_inv = evaluate_model(model_instance, X_test, y_test, close_scaler)
    test_start_idx = split_idx + seq_length
    test_dates = data_df.index[test_start_idx:].tolist()
    test_dates_str = [str(d.date()) for d in test_dates]
    return {
        "ticker": ticker,
        "test_dates": test_dates_str,
        "actual_test": y_test_inv.flatten().tolist(),
        "predicted_test": y_pred_inv.flatten().tolist(),
        "accuracy": metrics
    }

def predict_future_stream(ticker, future_days=10, model_path="saved_lstm_model.h5"):
    data_df = fetch_stock_data(ticker, period="10y")
    if isinstance(data_df, dict) and "error" in data_df:
        yield data_df
        return
    scaled_data, scaler, close_scaler = preprocess_data(data_df)
    X, y = create_sequences(scaled_data, seq_length=10)
    if len(X) == 0:
        yield {"error": "Not enough data after creating sequences."}
        return
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    model_instance = train_or_load_model(X_train, y_train, model_path)
    seq_length = 10
    if len(scaled_data) < seq_length:
        yield {"error": f"Insufficient data: have {len(scaled_data)}, need >= {seq_length}"}
        return

    # Get last trading day from dataset as starting point
    last_date = data_df.index[-1].date()
    days_generated = 0
    last_sequence = scaled_data[-seq_length:]
    while days_generated < future_days:
        # Increment one calendar day
        last_date += timedelta(days=1)
        # Skip weekends (Saturday=5, Sunday=6)
        if last_date.weekday() >= 5:
            continue
        pred = model_instance.predict(last_sequence.reshape(1, seq_length, scaled_data.shape[1]))
        # Use dedicated close scaler to inverse transform the prediction
        inv_pred = close_scaler.inverse_transform(pred)[0, 0]
        time.sleep(1)  # simulate real-time delay
        yield {
            "date": str(last_date),
            "prediction": float(inv_pred)
        }
        # Autoregressive update: note that feeding back predictions may accumulate errors.
        new_row = np.array([pred[0][0]] + [0]*(scaled_data.shape[1]-1))
        last_sequence = np.vstack([last_sequence[1:], new_row])
        days_generated += 1

def stock_price_prediction(ticker, future_days=10):
    data_df = fetch_stock_data(ticker, period="10y")
    if isinstance(data_df, dict) and "error" in data_df:
        return data_df
    historical_df = data_df.iloc[-30:].copy()
    historical_dates = [str(d.date()) for d in historical_df.index]
    historical_closes = historical_df["Close"].tolist()
    scaled_data, scaler, close_scaler = preprocess_data(data_df)
    X, y = create_sequences(scaled_data, seq_length=10)
    if len(X) == 0:
        return {"error": "Not enough data after creating sequences."}
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    model_instance = train_or_load_model(X_train, y_train)
    future_preds = []
    last_sequence = scaled_data[-10:]
    for _ in range(future_days):
        pred = model_instance.predict(last_sequence.reshape(1, 10, scaled_data.shape[1]))
        future_preds.append(pred[0][0])
        new_row = np.array([pred[0][0]] + [0]*(scaled_data.shape[1]-1))
        last_sequence = np.vstack([last_sequence[1:], new_row])
    # Inverse transform predictions using the dedicated close scaler
    future_preds_arr = np.array(future_preds).reshape(-1, 1)
    inv_future = close_scaler.inverse_transform(future_preds_arr).flatten()
    current_date = data_df.index[-1].date()
    future_dates = []
    while len(future_dates) < future_days:
        current_date += timedelta(days=1)
        if current_date.weekday() >= 5:
            continue
        future_dates.append(str(current_date))
    return {
        "ticker": ticker,
        "historical": {
            "dates": historical_dates,
            "closes": historical_closes
        },
        "predictions": inv_future.tolist(),
        "future_dates": future_dates
    }
