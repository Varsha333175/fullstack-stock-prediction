from flask import Flask, request, jsonify
from flask_cors import CORS
import model  # The updated model.py above

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Stock Prediction API Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        ticker = data.get("ticker", "AAPL").upper()
        # Convert 'days' to int to avoid str -> int errors
        future_days_raw = data.get("days", 10)
        try:
            future_days = int(future_days_raw)
        except ValueError:
            return jsonify({"error": f"'days' must be an integer, got {future_days_raw}"}), 400

        result = model.stock_price_prediction(ticker, future_days)
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        print("[ERROR] Exception in /predict route:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
