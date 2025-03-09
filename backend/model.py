import os
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    # Require at least 30 rows for LSTM
    if data.shape[0] < 30:
        return {"error": f"Not enough historical data for {ticker}. Only {data.shape[0]} records available."}

    # Technical indicators
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["RSI"] = 100 - (100 / (1 + (data["Close"].diff().clip(lower=0).rolling(14).mean() /
                                     data["Close"].diff().clip(upper=0).abs().rolling(14).mean())))

    data.dropna(inplace=True)

    if data.shape[0] < 20:
        return {"error": f"Not enough processed data for {ticker}. Only {data.shape[0]} records after cleaning."}

    return data

def preprocess_data(data):
    features = ["Close", "SMA_10", "EMA_10", "RSI", "Volume"]

    for feature in features:
        if feature not in data.columns:
            data[feature] = 0

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    return scaled_data, scaler

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])  # Predict the "Close" price (index 0)
    return np.array(X), np.array(y)

def train_or_load_model(X_train, y_train, model_path="saved_lstm_model.h5"):
    if os.path.exists(model_path):
        print("[INFO] Loading existing model from disk...")
        model = keras.models.load_model(model_path)
        # ✅ Re-compile to fix the warning & ensure consistent settings
        model.compile(optimizer="adam", loss="mean_squared_error")
    else:
        print("[INFO] Training a new LSTM model...")
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(64),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
        model.save(model_path)
    return model

def predict_future(model, data, scaler, future_days=10):
    print(f"[DEBUG] data.shape: {data.shape}")
    print(f"[DEBUG] model input shape (should be (None, 10, {data.shape[1]})): {model.input_shape}")

    seq_length = 10
    if len(data) < seq_length:
        return {"error": f"Insufficient data: only {len(data)} rows, need at least {seq_length}."}

    predictions = []
    last_sequence = data[-seq_length:]

    for i in range(future_days):
        print(f"[DEBUG] Iteration {i+1}, last_sequence.shape = {last_sequence.shape}")

        if last_sequence.shape != (seq_length, data.shape[1]):
            err = f"Shape mismatch: expected (10,{data.shape[1]}), got {last_sequence.shape}"
            print("[ERROR]", err)
            return {"error": err}

        try:
            pred = model.predict(last_sequence.reshape(1, seq_length, data.shape[1]))
        except Exception as e:
            print("[ERROR] Exception during model.predict():", e)
            return {"error": str(e)}

        predictions.append(pred[0][0])
        # Add new row with predicted "Close" in index 0, rest are 0
        new_row = np.array([pred[0][0]] + [0]*(data.shape[1]-1))
        # Shift
        last_sequence = np.vstack([last_sequence[1:], new_row])

    # Convert predictions to original scale
    replicate_cols = np.column_stack([predictions]*data.shape[1])
    inv_scale = scaler.inverse_transform(replicate_cols)[:, 0]
    return inv_scale

def stock_price_prediction(ticker, future_days=10):
    data_df = fetch_stock_data(ticker)
    if isinstance(data_df, dict) and "error" in data_df:
        return data_df  # pass error up

    # Keep last 30 days for frontend
    historical_df = data_df.iloc[-30:].copy()
    historical_dates = [str(x.date()) for x in historical_df.index]
    historical_closes = historical_df["Close"].tolist()

    scaled_data, scaler = preprocess_data(data_df)
    X, y = create_sequences(scaled_data, seq_length=10)
    if len(X) == 0:
        return {"error": "Not enough data after creating sequences."}

    split_idx = int(len(X)*0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]

    model = train_or_load_model(X_train, y_train)

    predicted_future = predict_future(model, scaled_data, scaler, future_days)
    if isinstance(predicted_future, dict) and "error" in predicted_future:
        return predicted_future

    # Generate future dates
    from datetime import timedelta
    last_date = data_df.index[-1].date()
    future_dates = []
    current_date = last_date
    for _ in range(future_days):
        current_date += timedelta(days=1)
        future_dates.append(str(current_date))

    return {
        "ticker": ticker,
        "historical": {
            "dates": historical_dates,
            "closes": historical_closes
        },
        "predictions": predicted_future.tolist(),
        "future_dates": future_dates
    }
