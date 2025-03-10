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
  const [testActual, setTestActual] = useState([]);
  const [testPredicted, setTestPredicted] = useState([]);
  const [accuracy, setAccuracy] = useState(null);
  const [futurePredictions, setFuturePredictions] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    socket.on("connect", () => console.log("[Socket] Connected:", socket.id));

    socket.on("backtest_result", (res) => {
      console.log("[Socket] backtest_result:", res);
      setTestDates(res.test_dates || []);
      setTestActual(res.actual_test || []);
      setTestPredicted(res.predicted_test || []);
      setAccuracy(res.accuracy || null);
    });

    socket.on("partial_future", (msg) => {
      console.log("[Socket] partial_future:", msg);
      setFuturePredictions((prev) => [...prev, { date: msg.date, value: msg.prediction }]);
    });

    socket.on("prediction_complete", () => setLoading(false));

    socket.on("error", (err) => {
      console.error("[Socket] error:", err);
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
    setTestActual([]);
    setTestPredicted([]);
    setAccuracy(null);
    setFuturePredictions([]);
    socket.emit("start_prediction", { ticker: ticker.toUpperCase(), days: parseInt(days, 10) });
  };

  return (
    <div className="bg-light min-vh-100 d-flex flex-column">
      <nav className="navbar navbar-dark bg-dark">
        <div className="container">
          <span className="navbar-brand">Stock Predictor</span>
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

        {accuracy && (
          <div className="text-center">
            <h4>Backtest Accuracy</h4>
            <div className="d-inline-block">
              {Object.entries(accuracy).map(([key, val]) => (
                <span key={key} className="badge bg-success me-2">{`${key}: ${val}`}</span>
              ))}
            </div>
          </div>
        )}

        {testDates.length > 0 && (
          <div className="mb-5" style={{ height: "400px" }}>
            <Line data={{ labels: testDates, datasets: [{ label: "Actual", data: testActual, borderColor: "green" }, { label: "Predicted", data: testPredicted, borderColor: "blue" }] }} />
          </div>
        )}

        {futurePredictions.length > 0 && (
          <div className="mb-5" style={{ height: "400px" }}>
            <Line data={{ labels: futurePredictions.map((p) => p.date), datasets: [{ label: `Predictions (${days} days)`, data: futurePredictions.map((p) => p.value), borderColor: "red" }] }} />
          </div>
        )}
      </div>

      <footer className="bg-dark text-white text-center py-3">
        <p className="mb-0">© 2025 Stock Predictor</p>
      </footer>
    </div>
  );
};

export default App;
