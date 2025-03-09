from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import model
from datetime import timedelta

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def home():
    return {"message": "Backtest + Real-Time LSTM with Advanced Indicators"}

@app.route("/predict", methods=["POST"])
def predict_rest():
    """
    Fallback REST endpoint for testing in Postman.
    Expects JSON: { "ticker": "GOOGL", "days": 16 }
    Returns a combined JSON with backtest and full future predictions.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400
        ticker = data.get("ticker", "AAPL").upper()
        days_raw = data.get("days", 10)
        try:
            future_days = int(days_raw)
        except:
            return jsonify({"error": f"'days' must be an integer, got {days_raw}"}), 400
        backtest_res = model.backtest_prediction(ticker)
        if isinstance(backtest_res, dict) and "error" in backtest_res:
            return jsonify(backtest_res), 400
        future_gen = model.predict_future_stream(ticker, future_days)
        future_list = []
        for val in future_gen:
            if isinstance(val, dict) and "error" in val:
                return jsonify(val), 400
            future_list.append(val)
        final_json = {
            "backtest_result": backtest_res,
            "future_prediction": future_list
        }
        return jsonify(final_json), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on("start_prediction")
def handle_start_prediction(data):
    try:
        ticker = data.get("ticker", "AAPL").upper()
        future_days = int(data.get("days", 10))
        backtest_res = model.backtest_prediction(ticker)
        if isinstance(backtest_res, dict) and "error" in backtest_res:
            emit("error", backtest_res)
            return
        emit("backtest_result", backtest_res)
        future_gen = model.predict_future_stream(ticker, future_days)
        day_index = 0
        for val in future_gen:
            if isinstance(val, dict) and "error" in val:
                emit("error", val)
                return
            day_index += 1
            emit("partial_future", {
                "day_index": day_index,
                "date": val["date"],
                "prediction": val["prediction"]
            })
        emit("prediction_complete", {"message": "All tasks done."})
    except Exception as e:
        emit("error", {"error": str(e)})

if __name__ == "__main__":
    socketio.run(app, debug=True)
