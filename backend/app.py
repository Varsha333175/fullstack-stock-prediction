from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from model import predict_future  # Ensure this is correctly imported

app = Flask(__name__)
CORS(app)

# ✅ Enable Logging
logging.basicConfig(level=logging.INFO)

@app.route("/")
def home():
    return jsonify({"message": "Stock Prediction API Running"})

@app.route("/predict", methods=["POST"])
def predict_stock():
    """REST API for stock prediction."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        ticker = data.get("ticker", "AAPL").upper()
        future_days = int(data.get("days", 10))

        # ✅ Log request details
        logging.info(f"📡 Received Prediction Request: {ticker}, Days: {future_days}")

        # ✅ Run prediction
        future_predictions = predict_future(ticker, future_days)

        # ✅ Log output before sending
        logging.info(f"📊 Future Predictions: {future_predictions}")

        return jsonify({"future_prediction": future_predictions}), 200

    except Exception as e:
        logging.error(f"❌ API Error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
