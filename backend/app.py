from flask import Flask, request, jsonify
from flask_cors import CORS
from model import train_model, predict_future, backtest_model

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Stock Prediction API Running"})

@app.route("/predict", methods=["POST"])
def predict_stock():
    """REST API for stock prediction and backtesting."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        ticker = data.get("ticker", "AAPL").upper()
        future_days = int(data.get("days", 10))

        # Predict future stock prices
        future_predictions = predict_future(ticker, future_days)

        # Backtest results
        backtest_results = backtest_model(ticker)

        return jsonify({
            "backtest_result": backtest_results,
            "future_prediction": future_predictions
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
