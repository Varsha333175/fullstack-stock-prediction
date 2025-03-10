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

        backtest_res = model.backtest_prediction(ticker)
        if "error" in backtest_res:
            return jsonify(backtest_res), 400

        future_predictions = model.predict_future_stream(ticker, future_days)

        return jsonify({"backtest_result": backtest_res, "future_prediction": future_predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on("start_prediction")
def handle_start_prediction(data):
    try:
        ticker = data.get("ticker", "AAPL").upper()
        future_days = int(data.get("days", 10))

        backtest_res = model.backtest_prediction(ticker)
        if "error" in backtest_res:
            emit("error", backtest_res)
            return

        emit("backtest_result", backtest_res)

        for val in model.predict_future_stream(ticker, future_days):
            if "error" in val:
                emit("error", val)
                return
            emit("partial_future", {"date": val["date"], "prediction": val["prediction"]})

        emit("prediction_complete", {"message": "Prediction process completed."})
    except Exception as e:
        emit("error", {"error": str(e)})

if __name__ == "__main__":
    print("Registered Routes:", [rule.rule for rule in app.url_map.iter_rules()])
    socketio.run(app, debug=True)
