import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import tensorflow as tf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, BatchNormalization, LSTM, GRU, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import optuna

# Download NLP model for sentiment analysis
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Logging Setup
logging.basicConfig(filename="stock_prediction.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Caching stock data
cache = {}

def fetch_stock_data_cached(ticker, period="5y"):
    """Fetches stock historical data and computes technical indicators."""
    if ticker in cache:
        return cache[ticker]

    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    if data.shape[0] < 200:
        return {"error": f"Not enough historical data for {ticker} ({data.shape[0]} records)."}

    data = data.interpolate(method='linear')
    data.dropna(inplace=True)

    # Use last 500 records for better accuracy
    data = data.iloc[-500:]

    # Technical Indicators
    data["SMA_20"] = data["Close"].rolling(20, min_periods=5).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["Momentum"] = data["Close"].diff(5).fillna(0)

    # RSI Calculation
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=5).mean()
    avg_loss = loss.rolling(14, min_periods=5).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    data["RSI"] = 100 - (100 / (1 + rs))

    # Add Market Sentiment Score
    data["Sentiment"] = fetch_stock_sentiment(ticker)

    data.dropna(inplace=True)

    if data.shape[0] < 100:
        return {"error": f"Not enough processed data for {ticker}, only {data.shape[0]} rows after cleaning."}

    cache[ticker] = data
    return data

def fetch_stock_sentiment(ticker):
    """Fetch stock news and analyze sentiment using Yahoo Finance API."""
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=5"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"⚠️ Yahoo Finance API Error: {response.status_code}. Using default sentiment.")
            return 0

        articles = response.json().get("news", [])
        if not articles:
            print("⚠️ No news articles found. Default sentiment applied.")
            return 0

        sentiment_scores = [sia.polarity_scores(article.get("title", ""))["compound"] for article in articles]
        return np.mean(sentiment_scores) if sentiment_scores else 0
    except Exception as e:
        print(f"⚠️ Sentiment Fetch Error: {str(e)}. Using default sentiment.")
        return 0

def preprocess_data(data):
    features = ["Open", "Close", "SMA_20", "EMA_20", "Momentum", "RSI", "Sentiment"]

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
        y_open.append(scaled_data[i+seq_length][0])
        y_close.append(scaled_data[i+seq_length][1])
    return np.array(X), np.array(y_open), np.array(y_close)

import optuna

def objective(trial, X_train, y_open_train, y_close_train):
    """
    Optuna hyperparameter tuning function to optimize the LSTM model.
    """
    seq_length = X_train.shape[1]
    feature_size = X_train.shape[2]

    # Suggest optimal values for model layers
    filters = trial.suggest_int("filters", 64, 256)
    lstm_units = trial.suggest_int("lstm_units", 128, 512)
    gru_units = trial.suggest_int("gru_units", 64, 256)
    dense_units = trial.suggest_int("dense_units", 64, 256)

    # Define the model with suggested hyperparameters
    model = Sequential([
        Input(shape=(seq_length, feature_size)),
        Conv1D(filters=filters, kernel_size=3, activation="relu"),
        BatchNormalization(),
        LSTM(lstm_units, return_sequences=True),
        BatchNormalization(),
        GRU(gru_units, return_sequences=False),
        Flatten(),
        Dense(dense_units, activation="relu"),
        Dense(2)  # Predict Open & Close
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model for a few epochs to evaluate performance
    history = model.fit(
        X_train, np.column_stack((y_open_train, y_close_train)),
        epochs=5, batch_size=32, verbose=0, validation_split=0.2
    )

    return history.history["val_loss"][-1]  # Optuna minimizes validation loss


def build_model(seq_length, feature_size):
    model = Sequential([
        Input(shape=(seq_length, feature_size)),
        Conv1D(filters=128, kernel_size=3, activation="relu"),
        BatchNormalization(),
        LSTM(256, return_sequences=True),
        BatchNormalization(),
        GRU(128, return_sequences=False),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(2)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_or_load_model(X_train, y_open_train, y_close_train, seq_length=60, model_path="models/stock_model_v6.keras"):
    """Trains or loads a model with optimized parameters."""
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        logging.info("Loaded pre-trained model.")
        return model

    print("🔍 Running Optuna Hyperparameter Optimization (Reduced Trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_open_train, y_close_train), n_trials=5)

    best_params = study.best_params
    print(f"✅ Best Hyperparameters: {best_params}")

    model = Sequential([
        Input(shape=(seq_length, 7)),  # Fix input shape issue
        Conv1D(filters=best_params["filters"], kernel_size=3, activation="relu"),
        BatchNormalization(),
        LSTM(best_params["lstm_units"], return_sequences=True),
        BatchNormalization(),
        GRU(best_params["gru_units"], return_sequences=False),
        Flatten(),
        Dense(best_params["dense_units"], activation="relu"),
        Dense(2)  # Predicts Open & Close prices
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    print("🚀 Training Model with Optimized Parameters...")
    model.fit(X_train, np.column_stack((y_open_train, y_close_train)), epochs=100, batch_size=32, verbose=1)

    model.save(model_path)
    return model

def predict_future_stream(ticker, future_days=30, model_path="models/stock_model_v6.keras"):
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

    for _ in range(future_days):
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

        new_row = last_sequence[-1]
        new_row[0] = pred_scaled[0][0]
        new_row[1] = pred_scaled[0][1]
        last_sequence = np.vstack([last_sequence[1:], new_row])

    return predictions
def backtest_prediction(ticker):
    """
    Backtests the model by comparing past predictions to actual stock prices.
    It returns accuracy metrics and the test dataset for graphing.
    """
    data_df = fetch_stock_data_cached(ticker, period="4y")  # Use 4 months for accuracy tracking
    if isinstance(data_df, dict) and "error" in data_df:
        return data_df

    scaled_data, _, open_scaler, close_scaler = preprocess_data(data_df)
    seq_length = 60
    X, y_open, y_close = create_sequences(scaled_data, seq_length)

    if len(X) == 0:
        return {"error": "Not enough data after creating sequences."}

    split_idx = int(len(X) * 0.8)
    X_train, y_open_train, y_close_train = X[:split_idx], y_open[:split_idx], y_close[:split_idx]
    X_test, y_open_test, y_close_test = X[split_idx:], y_open[split_idx:], y_close[split_idx:]

    model_instance = train_or_load_model(X_train, y_open_train, y_close_train, seq_length)
    
    # Predict past test data
    y_pred_scaled = model_instance.predict(X_test)
    y_pred_open_inv = open_scaler.inverse_transform(y_pred_scaled[:, 0].reshape(-1, 1)).flatten()
    y_pred_close_inv = close_scaler.inverse_transform(y_pred_scaled[:, 1].reshape(-1, 1)).flatten()

    y_open_test_inv = open_scaler.inverse_transform(y_open_test.reshape(-1, 1)).flatten()
    y_close_test_inv = close_scaler.inverse_transform(y_close_test.reshape(-1, 1)).flatten()

    # Calculate accuracy metrics
    mse_open = mean_squared_error(y_open_test_inv, y_pred_open_inv)
    mse_close = mean_squared_error(y_close_test_inv, y_pred_close_inv)
    rmse_open = np.sqrt(mse_open)
    rmse_close = np.sqrt(mse_close)

    # Extract past test predictions for graph
    test_dates = data_df.index[split_idx + seq_length:].tolist()
    test_dates_str = [str(d.date()) for d in test_dates]

    return {
        "ticker": ticker,
        "test_dates": test_dates_str,
        "actual_open_test": y_open_test_inv.tolist(),
        "predicted_open_test": y_pred_open_inv.tolist(),
        "actual_close_test": y_close_test_inv.tolist(),
        "predicted_close_test": y_pred_close_inv.tolist(),
        "accuracy": {
            "MSE Open": round(mse_open, 4),
            "MSE Close": round(mse_close, 4),
            "RMSE Open": round(rmse_open, 4),
            "RMSE Close": round(rmse_close, 4),
        }
    }
