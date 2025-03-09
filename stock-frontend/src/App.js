import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { io } from "socket.io-client";
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
import { Line } from "react-chartjs-2";

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

  // Backtest states
  const [testDates, setTestDates] = useState([]);
  const [testActual, setTestActual] = useState([]);
  const [testPredicted, setTestPredicted] = useState([]);
  const [accuracy, setAccuracy] = useState(null);

  // Future predictions (partial streaming)
  const [futurePredictions, setFuturePredictions] = useState([]);

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    socket.on("connect", () => {
      console.log("[Socket] Connected:", socket.id);
    });
    socket.on("backtest_result", (res) => {
      console.log("[Socket] backtest_result:", res);
      setTestDates(res.test_dates);
      setTestActual(res.actual_test);
      setTestPredicted(res.predicted_test);
      setAccuracy(res.accuracy);
    });
    socket.on("partial_future", (msg) => {
      console.log("[Socket] partial_future:", msg);
      setFuturePredictions((prev) => [...prev, { date: msg.date, value: msg.prediction }]);
    });
    socket.on("prediction_complete", (msg) => {
      console.log("[Socket] prediction_complete:", msg);
      setLoading(false);
    });
    socket.on("error", (err) => {
      console.log("[Socket] error:", err);
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
    socket.emit("start_prediction", {
      ticker: ticker.toUpperCase(),
      days: parseInt(days, 10)
    });
  };

  // Backtest Chart Data
  const backtestData = {
    labels: testDates,
    datasets: [
      {
        label: "Actual (Test)",
        data: testActual,
        borderColor: "green",
        backgroundColor: "rgba(0,255,0,0.2)",
        fill: true
      },
      {
        label: "Predicted (Test)",
        data: testPredicted,
        borderColor: "blue",
        backgroundColor: "rgba(0,0,255,0.2)",
        fill: true
      }
    ]
  };
  const backtestOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: { enabled: true },
      legend: { display: true, position: "top" },
      title: { display: true, text: "Backtest: Actual vs. Predicted" }
    },
    scales: {
      x: { title: { display: true, text: "Dates" } },
      y: {
        title: { display: true, text: "Stock Price (USD)" },
        ticks: { callback: (val) => `$${val.toFixed(2)}` }
      }
    }
  };

  // Future Motion Chart Data
  const futureLabels = futurePredictions.map((p) => p.date);
  const futureValues = futurePredictions.map((p) => p.value);
  const futureData = {
    labels: futureLabels,
    datasets: [
      {
        label: `Future Predictions (Next ${days} Trading Days)`,
        data: futureValues,
        borderColor: "red",
        backgroundColor: "rgba(255,0,0,0.2)",
        fill: true
      }
    ]
  };
  const futureOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: { enabled: true },
      legend: { display: true, position: "top" },
      title: { display: true, text: "Future Motion Graph (Skipping Weekends)" }
    },
    scales: {
      x: { title: { display: true, text: "Dates" } },
      y: {
        title: { display: true, text: "Stock Price (USD)" },
        ticks: { callback: (val) => `$${val.toFixed(2)}` }
      }
    }
  };

  return (
    <div className="bg-light min-vh-100 d-flex flex-column">
      <nav className="navbar navbar-dark bg-dark">
        <div className="container">
          <span className="navbar-brand">Advanced LSTM Stock Predictor</span>
        </div>
      </nav>

      <div className="container my-4 flex-grow-1">
        <div className="row mb-3">
          <div className="col text-center">
            <h1>Predict {days} Future Trading Days for {ticker || "???"}</h1>
            <p className="text-muted">
              Backtested accuracy is shown, and future predictions update in real time!
            </p>
          </div>
        </div>

        <div className="row justify-content-center mb-4">
          <div className="col-md-6">
            <div className="card p-3">
              <label className="form-label">Stock Ticker</label>
              <input
                type="text"
                className="form-control mb-2"
                placeholder="e.g., AAPL"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
              />
              <label className="form-label">Future Days</label>
              <input
                type="number"
                className="form-control mb-3"
                placeholder="10"
                value={days}
                onChange={(e) => setDays(e.target.value)}
              />
              <button className="btn btn-primary" onClick={startPrediction} disabled={loading}>
                {loading ? "Predicting..." : "Predict"}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="row mb-3">
            <div className="col text-center">
              <div className="alert alert-danger">{error}</div>
            </div>
          </div>
        )}

        {accuracy && (
          <div className="row mb-3">
            <div className="col text-center">
              <h4>Backtest Accuracy</h4>
              <div className="d-inline-block">
                <span className="badge bg-success me-2">MSE: {accuracy.MSE}</span>
                <span className="badge bg-success me-2">RMSE: {accuracy.RMSE}</span>
                <span className="badge bg-success me-2">MAPE: {accuracy.MAPE}</span>
                <span className="badge bg-success me-2">R²: {accuracy.R2}</span>
              </div>
            </div>
          </div>
        )}

        {testDates.length > 0 && (
          <div className="row mb-5">
            <div className="col" style={{ height: "400px" }}>
              <Line data={backtestData} options={backtestOptions} />
            </div>
          </div>
        )}

        {futurePredictions.length > 0 && (
          <div className="row mb-5">
            <div className="col" style={{ height: "400px" }}>
              <Line data={futureData} options={futureOptions} />
            </div>
          </div>
        )}
      </div>

      <footer className="bg-dark text-white text-center py-3">
        <div className="container">
          <p className="mb-0">© 2025 Advanced LSTM Stock Predictor (Skipping Weekends)</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
