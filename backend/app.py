from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import model

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def home():
    return {"message": "Stock Prediction API Running"}

@app.route("/predict", methods=["POST"])
def predict_rest():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        ticker = data.get("ticker", "AAPL").upper()
        future_days = int(data.get("days", 10))

        future_predictions = model.predict_future_stream(ticker, future_days)

        return jsonify({"future_prediction": future_predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    socketio.run(app, debug=True)
