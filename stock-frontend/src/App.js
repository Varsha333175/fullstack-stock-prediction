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

const App = () => {
  const [ticker, setTicker] = useState("");
  const [days, setDays] = useState(10);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const fetchPredictions = async () => {
    setLoading(true);
    setError("");
    setPredictions([]);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.toUpperCase(), days: parseInt(days, 10) }),
      });

      const data = await response.json();
      if (data.error) setError(data.error);
      else setPredictions(data.future_prediction);
    } catch (err) {
      setError("Failed to fetch predictions");
    }
    setLoading(false);
  };

  const chartData = {
    labels: predictions.map((p) => p.date),
    datasets: [
      {
        label: "Predicted Open Price",
        data: predictions.map((p) => p.open_prediction),
        borderColor: "green",
        backgroundColor: "rgba(0, 255, 0, 0.2)",
        fill: true,
      },
      {
        label: "Predicted Close Price",
        data: predictions.map((p) => p.close_prediction),
        borderColor: "blue",
        backgroundColor: "rgba(0, 0, 255, 0.2)",
        fill: true,
      },
    ],
  };
  
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true, position: "top" },
      title: { display: true, text: `Predicted Stock Prices for ${ticker.toUpperCase()}` },
    },
    scales: {
      x: { title: { display: true, text: "Date" } },
      y: {
        title: { display: true, text: "Stock Price (USD)" },
        ticks: { callback: (val) => `$${val.toFixed(2)}` },
      },
    },
  };

  return (
    <div className="bg-light min-vh-100 d-flex flex-column">
      <nav className="navbar navbar-dark bg-dark">
        <div className="container">
          <span className="navbar-brand">Stock Price Predictor</span>
        </div>
      </nav>

      <div className="container my-4">
        <div className="row justify-content-center">
          <div className="col-md-6">
            <div className="card p-3">
              <h4 className="text-center">Stock Prediction</h4>
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

        {error && <div className="alert alert-danger text-center mt-3">{error}</div>}

        {predictions.length > 0 && (
          <div className="mt-4">
            <h4 className="text-center">Stock Price Prediction Graph</h4>
            <div className="chart-container" style={{ height: "400px" }}>
              <Line data={chartData} options={chartOptions} />
            </div>
          </div>
        )}
      </div>

      <footer className="bg-dark text-white text-center py-3 mt-auto">
        <p className="mb-0">© 2025 Stock Predictor</p>
      </footer>
    </div>
  );
};

export default App;
