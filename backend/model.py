import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Function to fetch stock data
def fetch_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    if data.shape[0] < 30:  # Require at least 30 rows before processing
        return {"error": f"Not enough historical data for {ticker}. Only {data.shape[0]} records available."}

    # Calculate technical indicators
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["RSI"] = 100 - (100 / (1 + (data["Close"].diff().clip(lower=0).rolling(14).mean() /
                                     data["Close"].diff().clip(upper=0).abs().rolling(14).mean())))
    
    data["Volume"] = data["Volume"]
    data = data.dropna()

    if data.shape[0] < 20:  # Ensure at least 20 rows after cleaning
        return {"error": f"Not enough processed data for {ticker}. Only {data.shape[0]} records available after cleaning."}

    return data

# Function to preprocess data
def preprocess_data(data):
    features = ["Close", "SMA_10", "EMA_10", "RSI", "Volume"]

    # Ensure all features exist
    for feature in features:
        if feature not in data.columns:
            data[feature] = 0  

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    return scaled_data, scaler

# Function to create sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)

# Function to train LSTM model
def train_model(X_train, y_train):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=False),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    return model

# Function to predict future prices
def predict_future(model, data, scaler, future_days=10):
    predictions = []

    # Ensure we have at least 10 rows
    if len(data) < 10:
        return {"error": f"Insufficient data: Only {len(data)} records available, need at least 10."}

    last_sequence = data[-10:]

    print(f"Data shape before reshaping: {data.shape}")  
    print(f"Last sequence shape before reshaping: {last_sequence.shape}")

    for i in range(future_days):
        print(f"Iteration {i+1} - Last sequence shape: {last_sequence.shape}")  # Debug print

        if last_sequence.shape != (10, data.shape[1]):  
            print(f"Error: Mismatched shape at iteration {i+1}! Last sequence shape: {last_sequence.shape}")
            return {"error": f"Cannot reshape array of size {last_sequence.size} into shape (10,{data.shape[1]})"}

        prediction = model.predict(last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1]))
        predictions.append(prediction[0][0])

        # Ensure correct array appending
        last_sequence = np.vstack([last_sequence[1:], np.array([prediction[0][0]] * data.shape[1])])

    return scaler.inverse_transform(np.column_stack([predictions] * data.shape[1]))[:, 0]

# Main function
def stock_price_prediction(ticker):
    data = fetch_stock_data(ticker)
    if isinstance(data, dict) and "error" in data:
        return data  

    scaled_data, scaler = preprocess_data(data)
    X, y = create_sequences(scaled_data)

    if len(X) == 0:  # Prevent empty training sets
        return {"error": "Insufficient training data after sequence processing."}

    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]

    model = train_model(X_train, y_train)
    return predict_future(model, scaled_data, scaler)
