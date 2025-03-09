import React, { useState, useEffect } from "react";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css"; // Import Bootstrap
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
import { Line } from "react-chartjs-2";

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Title,
  Legend
);

const App = () => {
  const [ticker, setTicker] = useState("");
  const [days, setDays] = useState(10);
  const [historicalDates, setHistoricalDates] = useState([]);
  const [historicalCloses, setHistoricalCloses] = useState([]);
  const [futureDates, setFutureDates] = useState([]);
  const [predictions, setPredictions] = useState([]);
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
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        ticker: ticker.toUpperCase(),
        days: days
      });

      if (response.data && response.data.historical && response.data.predictions) {
        setHistoricalDates(response.data.historical.dates);
        setHistoricalCloses(response.data.historical.closes);
        setFutureDates(response.data.future_dates);
        setPredictions(response.data.predictions);
      } else if (response.data && response.data.error) {
        setError(response.data.error);
      } else {
        setError("Invalid response from server.");
      }
    } catch (err) {
      setError("Error fetching predictions. Is Flask API running?");
    }
    setLoading(false);
  };

  // Combine historical + future for a single timeline
  const allDates = [...historicalDates, ...futureDates];
  const historicalDataSeries = [
    ...historicalCloses,
    ...new Array(predictions.length).fill(null)
  ];
  const predictionDataSeries = [
    ...new Array(historicalCloses.length).fill(null),
    ...predictions
  ];

  const chartData = {
    labels: allDates,
    datasets: [
      {
        label: `Historical (${ticker.toUpperCase()})`,
        data: historicalDataSeries,
        borderColor: "green",
        backgroundColor: "rgba(0, 255, 0, 0.3)",
        fill: true
      },
      {
        label: `Predicted Next ${days} Days`,
        data: predictionDataSeries,
        borderColor: "blue",
        backgroundColor: "rgba(0, 0, 255, 0.3)",
        fill: true
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: {
        enabled: true,
        callbacks: {
          label: (tooltipItem) => {
            const val = tooltipItem.raw;
            if (val === null) return "";
            return `$${parseFloat(val).toFixed(2)}`;
          }
        }
      },
      legend: {
        display: true,
        position: "top"
      },
      title: {
        display: true,
        text: "Stock Price: Historical + Future Prediction"
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Date"
        },
        ticks: {
          maxTicksLimit: 10,
          autoSkip: true,
          autoSkipPadding: 5
        }
      },
      y: {
        title: {
          display: true,
          text: "Stock Price (USD)"
        },
        ticks: {
          callback: (value) => `$${value.toFixed(2)}`
        }
      }
    }
  };

  return (
    <div className="bg-light min-vh-100 d-flex flex-column">
      {/* NavBar */}
      <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
        <div className="container">
          <a className="navbar-brand" href="#home">
            My Stock Predictor
          </a>
        </div>
      </nav>

      {/* Main Content */}
      <div className="container flex-grow-1 my-4">
        <div className="row mb-3">
          <div className="col text-center">
            <h1>Professional Stock Prediction</h1>
            <p className="text-muted">
              Enter a stock ticker and see the last 30 days of history plus the next {days} days of predictions.
            </p>
          </div>
        </div>

        {/* Input Form */}
        <div className="row justify-content-center mb-4">
          <div className="col-md-6">
            <div className="card p-3">
              <div className="mb-3">
                <label htmlFor="tickerInput" className="form-label">
                  Stock Ticker
                </label>
                <input
                  type="text"
                  id="tickerInput"
                  className="form-control"
                  placeholder="e.g., AAPL"
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value)}
                />
              </div>
              <div className="mb-3">
                <label htmlFor="daysInput" className="form-label">
                  Future Days to Predict
                </label>
                <input
                  type="number"
                  id="daysInput"
                  className="form-control"
                  placeholder="10"
                  value={days}
                  onChange={(e) => setDays(e.target.value)}
                />
              </div>
              <button
                className="btn btn-primary"
                onClick={fetchPredictions}
                disabled={loading}
              >
                {loading ? "Predicting..." : "Predict"}
              </button>
            </div>
          </div>
        </div>

        {/* Error or Chart */}
        {error && (
          <div className="row justify-content-center mb-4">
            <div className="col-md-6">
              <div className="alert alert-danger text-center">{error}</div>
            </div>
          </div>
        )}

        {(historicalDates.length > 0 || predictions.length > 0) && !error && (
          <div className="row">
            <div className="col-12" style={{ height: "500px" }}>
              <Line data={chartData} options={chartOptions} />
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-dark text-white text-center py-3 mt-auto">
        <div className="container">
          <p className="mb-0">© 2025 My Stock Predictor - All Rights Reserved</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
