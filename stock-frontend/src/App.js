import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Title,
  Legend
} from "chart.js";

// Register Chart.js components
ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Title, Legend);

const API_URL = "http://127.0.0.1:5000/predict"; // Flask API URL

const App = () => {
  const [ticker, setTicker] = useState("");
  const [days, setDays] = useState(10);
  const [testDates, setTestDates] = useState([]);
  const [testActualClose, setTestActualClose] = useState([]);
  const [testPredictedClose, setTestPredictedClose] = useState([]);
  const [futureDates, setFutureDates] = useState([]);
  const [futureOpen, setFutureOpen] = useState([]);
  const [futureClose, setFutureClose] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const fetchPredictions = async () => {
    if (!ticker) {
      setError("Please enter a stock ticker.");
      return;
    }

    setError("");
    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.toUpperCase(), days: parseInt(days, 10) })
      });

      const data = await response.json();

      if (response.ok) {
        console.log("API Response:", data);

        // Backtest results
        setTestDates(data.backtest_result.test_dates || []);
        setTestActualClose(data.backtest_result.actual_close_prices || []);
        setTestPredictedClose(data.backtest_result.predicted_close_prices || []);

        // Future predictions
        setFutureDates(data.future_prediction.map((p) => p.date) || []);
        setFutureOpen(data.future_prediction.map((p) => p.open_prediction) || []);
        setFutureClose(data.future_prediction.map((p) => p.close_prediction) || []);
      } else {
        setError(data.error || "Error fetching predictions.");
      }
    } catch (err) {
      setError("Network error: Unable to fetch data.");
      console.error("Fetch error:", err);
    } finally {
      setLoading(false);
    }
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
              <button className="btn btn-primary" onClick={fetchPredictions} disabled={loading}>
                {loading ? "Predicting..." : "Predict"}
              </button>
            </div>
          </div>
        </div>

        {error && <div className="alert alert-danger text-center">{error}</div>}

        {/* Backtest Results Graph */}
        {testDates.length > 0 && testActualClose.length > 0 && testPredictedClose.length > 0 && (
          <div className="mb-5" style={{ height: "400px" }}>
            <Line
              data={{
                labels: testDates,
                datasets: [
                  { label: "Actual Close", data: testActualClose, borderColor: "purple", borderWidth: 2 },
                  { label: "Predicted Close", data: testPredictedClose, borderColor: "red", borderWidth: 2, borderDash: [5, 5] }
                ]
              }}
              options={{
                scales: {
                  y: {
                    beginAtZero: false,
                    ticks: {
                      callback: function (value) {
                        return "$" + value.toFixed(2); // Format as USD
                      }
                    }
                  }
                },
                plugins: {
                  legend: { position: "top" },
                  tooltip: {
                    callbacks: {
                      label: function (tooltipItem) {
                        return tooltipItem.dataset.label + ": $" + tooltipItem.raw.toFixed(2);
                      }
                    }
                  }
                }
              }}
            />
          </div>
        )}

        {/* Future Predictions Graph */}
        {futureDates.length > 0 && futureOpen.length > 0 && futureClose.length > 0 && (
          <div className="mb-5" style={{ height: "400px" }}>
            <Line
              data={{
                labels: futureDates,
                datasets: [
                  { label: "Predicted Open", data: futureOpen, borderColor: "blue", borderWidth: 2 },
                  { label: "Predicted Close", data: futureClose, borderColor: "red", borderWidth: 2, borderDash: [5, 5] }
                ]
              }}
              options={{
                scales: {
                  y: {
                    beginAtZero: false,
                    ticks: {
                      callback: function (value) {
                        return "$" + value.toFixed(2); // Format as USD
                      }
                    }
                  }
                }
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
