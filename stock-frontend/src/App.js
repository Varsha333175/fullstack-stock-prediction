import React, { useState, useEffect } from "react";
import axios from "axios";
import { Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Title, Legend } from "chart.js";
import { Line } from "react-chartjs-2";

// Register Chart.js components
ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Title, Legend);

const App = () => {
  const [ticker, setTicker] = useState(""); // User input stock ticker
  const [predictions, setPredictions] = useState([]); // Predicted stock prices
  const [loading, setLoading] = useState(false); // Show loading state
  const [error, setError] = useState(""); // Error message

  const fetchPredictions = async () => {
    if (!ticker) {
      setError("Please enter a stock ticker.");
      return;
    }

    setLoading(true);
    setError("");
    setPredictions([]); // Clear previous results

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        ticker: ticker.toUpperCase(),
      });

      console.log("API Response:", response.data); // Debugging print

      if (response.data && response.data.predictions) {
        let formattedPredictions = response.data.predictions.map((p) => parseFloat(p)); // Convert to float

        console.log("Formatted Predictions for Chart (Before Sorting):", formattedPredictions); // Debugging print

        // Ensure React updates state properly by spreading values
        setPredictions([...formattedPredictions]);
      } else {
        setError("Invalid API response.");
      }
    } catch (err) {
      setError("Error fetching predictions. Check if Flask API is running.");
    }

    setLoading(false);
  };

  useEffect(() => {
    console.log("Updated Predictions for Chart:", predictions); // Debugging print
  }, [predictions]);

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>📈 Stock Price Prediction</h1>
      <input
        type="text"
        placeholder="Enter Stock Ticker (e.g., AAPL)"
        value={ticker}
        onChange={(e) => setTicker(e.target.value)}
        style={{ padding: "10px", fontSize: "16px", marginRight: "10px" }}
      />
      <button onClick={fetchPredictions} style={{ padding: "10px", fontSize: "16px" }}>
        Predict
      </button>

      {loading && <p>Loading predictions...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {predictions.length > 0 && (
        <div style={{ width: "600px", height: "400px", margin: "auto", marginTop: "20px" }}>
          <h2>Predicted Prices for {ticker.toUpperCase()}</h2>
          <Line
            data={{
              labels: predictions.map((_, i) => `Day ${i + 1}`), // Ensure correct order
              datasets: [
                {
                  label: `Predicted Stock Price for ${ticker.toUpperCase()}`,
                  data: predictions,
                  borderColor: "blue",
                  backgroundColor: "rgba(0, 0, 255, 0.3)",
                  fill: true,
                },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                tooltip: {
                  enabled: true,
                  callbacks: {
                    label: (tooltipItem) => `$${tooltipItem.raw?.toFixed(2)}`,
                  },
                },
                legend: {
                  display: true,
                  position: "top",
                },
              },
              scales: {
                x: {
                  title: {
                    display: true,
                    text: "Future Days",
                  },
                  ticks: {
                    maxTicksLimit: 10,
                  },
                },
                y: {
                  title: {
                    display: true,
                    text: "Stock Price (USD)",
                  },
                  ticks: {
                    callback: (value) => `$${value.toFixed(2)}`,
                  },
                },
              },
            }}
          />
        </div>
      )}
    </div>
  );
};

export default App;
