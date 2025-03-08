from flask import Flask, request, jsonify
from flask_cors import CORS
import model  # Import our LSTM model

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

@app.route("/")
def home():
    return jsonify({"message": "Stock Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        stock_ticker = data.get("ticker", "AAPL")  # Default to AAPL if no input

        predictions = model.stock_price_prediction(stock_ticker)

        if isinstance(predictions, dict) and "error" in predictions:
            return jsonify(predictions)  # Return error if stock has insufficient data

        return jsonify({"ticker": stock_ticker, "predictions": predictions.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
