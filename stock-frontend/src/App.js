import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Line } from "react-chartjs-2";
import { motion } from "framer-motion";
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

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Title, Legend);

const API_URL = "http://127.0.0.1:5000/predict";

const App = () => {
  const [ticker, setTicker] = useState("");
  const [days, setDays] = useState(10);
  const [testDates, setTestDates] = useState([]);
  const [testActualClose, setTestActualClose] = useState([]);
  const [testPredictedClose, setTestPredictedClose] = useState([]);
  const [futureDates, setFutureDates] = useState([]);
  const [futureOpen, setFutureOpen] = useState([]);
  const [futureClose, setFutureClose] = useState([]);
  const [accuracyMetrics, setAccuracyMetrics] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const fetchPredictions = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.toUpperCase(), days: parseInt(days, 10) }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Error fetching predictions.");
      }

      console.log("API Response:", data);

      setFutureDates(data.future_prediction.map((p) => p.date) || []);
      setFutureOpen(data.future_prediction.map((p) => p.open_prediction) || []);
      setFutureClose(data.future_prediction.map((p) => p.close_prediction) || []);
      setAccuracyMetrics(data.accuracy_metrics);
    } catch (error) {
      console.error("Fetch error:", error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="bg-light min-vh-100 d-flex flex-column"
    >
      <nav className="navbar navbar-dark bg-dark">
        <div className="container">
          <span className="navbar-brand">Stock Price Predictor</span>
        </div>
      </nav>

      <div className="container my-4">
        <motion.div
          className="row justify-content-center mb-4"
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
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
        </motion.div>

        {error && <div className="alert alert-danger text-center">{error}</div>}

        {accuracyMetrics && (
          <div className="card p-3 my-4">
            <h4>Model Accuracy Metrics</h4>
            <ul>
              <li><strong>MAE:</strong> {accuracyMetrics.MAE}</li>
              <li><strong>MSE:</strong> {accuracyMetrics.MSE}</li>
              <li><strong>RMSE:</strong> {accuracyMetrics.RMSE}</li>
              <li><strong>R² Score:</strong> {accuracyMetrics["R2 Score"]}</li>
            </ul>
          </div>
        )}

        {futureDates.length > 0 && (
          <motion.div className="mb-5" style={{ height: "400px" }} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1 }}>
            <Line
              data={{
                labels: futureDates,
                datasets: [
                  { label: "Predicted Open", data: futureOpen, borderColor: "blue", borderWidth: 2 },
                  { label: "Predicted Close", data: futureClose, borderColor: "red", borderWidth: 2, borderDash: [5, 5] }
                ]
              }}
              options={{ responsive: true, plugins: { legend: { position: "top" } } }}
            />
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default App;
