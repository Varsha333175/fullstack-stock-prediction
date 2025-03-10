import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { io } from "socket.io-client";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Title,
  Legend,
  Filler
} from "chart.js";

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Title,
  Legend,
  Filler
);

const socket = io("http://127.0.0.1:5000");

const App = () => {
  const [ticker, setTicker] = useState("");
  const [days, setDays] = useState(10);
  const [testDates, setTestDates] = useState([]);
  const [testActualOpen, setTestActualOpen] = useState([]);
  const [testActualClose, setTestActualClose] = useState([]);
  const [testPredictedOpen, setTestPredictedOpen] = useState([]);
  const [testPredictedClose, setTestPredictedClose] = useState([]);
  const [futurePredictions, setFuturePredictions] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    socket.on("connect", () => console.log("[Socket] Connected:", socket.id));

    socket.on("backtest_result", (res) => {
      console.log("[Socket] Backtest Result:", res);
      setTestDates(res.test_dates || []);
      setTestActualOpen(res.actual_open_test || []);
      setTestActualClose(res.actual_close_test || []);
      setTestPredictedOpen(res.predicted_open_test || []);
      setTestPredictedClose(res.predicted_close_test || []);
    });

    socket.on("partial_future", (msg) => {
      console.log("[Socket] Partial Future:", msg);
      setFuturePredictions((prev) => [...prev, { date: msg.date, open: msg.open, close: msg.close }]);
    });

    socket.on("prediction_complete", () => setLoading(false));
    
    socket.on("error", (err) => {
      console.error("[Socket] Error:", err);
      setError(err.error || JSON.stringify(err));
      setLoading(false);
    });

    return () => {
      socket.off("connect");
      socket.off("backtest_result");
      socket.off("partial_future");
      socket.off("prediction_complete");
      socket.off("error");
    };
  }, []);

  const startPrediction = () => {
    if (!ticker) {
      setError("Please enter a stock ticker.");
      return;
    }
    setError("");
    setLoading(true);
    setTestDates([]);
    setTestActualOpen([]);
    setTestActualClose([]);
    setTestPredictedOpen([]);
    setTestPredictedClose([]);
    setFuturePredictions([]);
    socket.emit("start_prediction", { ticker: ticker.toUpperCase(), days: parseInt(days, 10) });
  };

  return (
    <div className="bg-light min-vh-100 d-flex flex-column">
      <nav className="navbar navbar-dark bg-dark">
        <div className="container">
          <span className="navbar-brand">Stock Price Predictor</span>
        </div>
      </nav>

      <div className="container my-4">
        <div className="row justify-content-center mb-4">
          <div className="col-md-6">
            <div className="card p-3">
              <label className="form-label">Stock Ticker</label>
              <input type="text" className="form-control mb-2" placeholder="e.g., AAPL" value={ticker} onChange={(e) => setTicker(e.target.value)} />
              <label className="form-label">Future Days</label>
              <input type="number" className="form-control mb-3" placeholder="10" value={days} onChange={(e) => setDays(e.target.value)} />
              <button className="btn btn-primary" onClick={startPrediction} disabled={loading}>
                {loading ? "Predicting..." : "Predict"}
              </button>
            </div>
          </div>
        </div>

        {error && <div className="alert alert-danger text-center">{error}</div>}

        {testDates.length > 0 && (
          <div className="mb-5" style={{ height: "400px" }}>
            <Line data={{
              labels: testDates,
              datasets: [
                { label: "Actual Open", data: testActualOpen, borderColor: "green" },
                { label: "Predicted Open", data: testPredictedOpen, borderColor: "blue" },
                { label: "Actual Close", data: testActualClose, borderColor: "purple" },
                { label: "Predicted Close", data: testPredictedClose, borderColor: "red" },
              ]
            }} />
          </div>
        )}

        {futurePredictions.length > 0 && (
          <div className="mb-5" style={{ height: "400px" }}>
            <Line data={{
              labels: futurePredictions.map((p) => p.date),
              datasets: [
                { label: "Predicted Open", data: futurePredictions.map((p) => p.open), borderColor: "blue" },
                { label: "Predicted Close", data: futurePredictions.map((p) => p.close), borderColor: "red" }
              ]
            }} />
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
