import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import logging
import requests
import yfinance as yf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# ✅ Enable Logging
logging.basicConfig(level=logging.INFO)

# ✅ Paths to Save Models
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pt")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    logging.info("📂 Created 'models/' directory")

# ✅ Define LSTM Model
class LightLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LightLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])

# ✅ Fetch Stock Data
def fetch_stock_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.shape[0] < 200:
        return {"error": f"Not enough historical data for {ticker}"}

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Momentum"] = df["Close"].diff(5).fillna(0)

    df.dropna(inplace=True)
    return df

# ✅ Fetch Market Sentiment Score
def fetch_sentiment_score(ticker):
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbol={ticker}&apikey=YOUR_API_KEY"
        response = requests.get(url).json()
        if "feed" not in response:
            return 0  
        sentiment_scores = [article["overall_sentiment_score"] for article in response["feed"]]
        return sum(sentiment_scores) / len(sentiment_scores)
    except:
        return 0  

# ✅ Fetch Fundamental Stock Data
def fetch_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", 0),
        "eps": info.get("trailingEps", 0),
        "dividend_yield": info.get("dividendYield", 0)
    }

# ✅ Fetch Macroeconomic Indicators
def fetch_macro_data():
    try:
        url = "https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key=YOUR_API_KEY&file_type=json"
        response = requests.get(url).json()
        interest_rate = float(response["observations"][-1]["value"])
        return {"interest_rate": interest_rate}
    except:
        return {"interest_rate": 0}

# ✅ Preprocess Data
def preprocess_data(df, ticker):
    fundamentals = fetch_fundamentals(ticker)
    macro_data = fetch_macro_data()

    df["Market_Cap"] = fundamentals["market_cap"]
    df["P/E_Ratio"] = fundamentals["pe_ratio"]
    df["EPS"] = fundamentals["eps"]
    df["Dividend_Yield"] = fundamentals["dividend_yield"]
    df["Interest_Rate"] = macro_data["interest_rate"]

    features = [
        "Open", "Close", "SMA_20", "EMA_20", "Momentum",
        "Market_Cap", "P/E_Ratio", "EPS", "Dividend_Yield",
        "Interest_Rate"
    ]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    return scaled_data, scaler

# ✅ Create Sequences
def create_sequences(data, seq_length=180):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][1])
    return np.array(X), np.array(y)

# ✅ Train or Load Model
import time

def train_hybrid_model(ticker):
    df = fetch_stock_data(ticker)
    if isinstance(df, dict) and "error" in df:
        return df

    scaled_data, scaler = preprocess_data(df, ticker)
    X, y_actual = create_sequences(scaled_data, seq_length=180)

    sentiment_score = fetch_sentiment_score(ticker)
    y_actual = [y * (1 + sentiment_score * 0.01) for y in y_actual]

    model = LightLSTM(input_dim=X.shape[2])
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X[:-60], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_actual[:-60], dtype=torch.float32).unsqueeze(1)

    for epoch in range(10):  
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            logging.info(f"🔄 Training Epoch [{epoch+1}/10] - Loss: {loss.item():.4f}")

    logging.info("✅ Model Retrained")

    # ✅ Return the trained model and scaler every time
    return model, scaler

def predict_future(ticker, days=15):
    """Predicts future stock prices while dynamically updating predictions."""
    logging.info(f"📢 Predicting {days} days for {ticker}")

    df = fetch_stock_data(ticker, period="2y")
    if isinstance(df, dict) and "error" in df:
        return df

    scaled_data, scaler = preprocess_data(df, ticker)
    X, _ = create_sequences(scaled_data, seq_length=180)
    
    # ✅ Ensure last_sequence has the correct shape (1, 180, 10)
    last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])

    model, _ = train_hybrid_model(ticker)

    predictions = []
    last_date = df.index[-1].date()
    sentiment_score = fetch_sentiment_score(ticker)

    volatility_factor_range = (0.99, 1.01)

    for _ in range(days):
        last_date += timedelta(days=1)
        if last_date.weekday() >= 5:
            continue

        X_tensor = torch.tensor(last_sequence, dtype=torch.float32)
        pred_scaled = model(X_tensor).detach().numpy().flatten()

        # ✅ Ensure predicted value is mapped to the correct feature index
        new_row = np.zeros((1, last_sequence.shape[2]))  # Ensure 10 features
        new_row[0, 1] = pred_scaled[0]  # Assign to the 'Close' feature column

        # ✅ Convert scaled prediction to actual price
        dummy_features = np.zeros((1, scaled_data.shape[1]))  
        dummy_features[:, 1] = pred_scaled[0]  
        pred_actual = scaler.inverse_transform(dummy_features)[:, 1][0]

        # ✅ Apply sentiment adjustment
        sentiment_adjustment = 1 + (sentiment_score * 0.001)
        volatility_factor = np.random.uniform(*volatility_factor_range)
        pred_actual *= volatility_factor * sentiment_adjustment

        # ✅ Prevent unrealistic sharp drops (>2% in one day)
        last_close_price = df["Close"].iloc[-1]
        if abs(pred_actual - last_close_price) > (last_close_price * 0.02):  
            pred_actual = last_close_price * np.random.uniform(0.99, 1.01)

        predictions.append({"date": str(last_date), "close_prediction": round(float(pred_actual), 2)})
        logging.info(f"📅 {last_date}: {pred_actual} USD")

        # ✅ Properly reshape and update last_sequence
        last_sequence = np.vstack([last_sequence[0][1:], new_row])  # (180, 10)
        last_sequence = last_sequence.reshape(1, 180, last_sequence.shape[1])  # Ensure correct shape (1,180,10)

    logging.info("✅ Prediction Completed")
    return predictions
