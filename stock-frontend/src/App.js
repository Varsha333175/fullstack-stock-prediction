import React, { useState } from "react";

const App = () => {
  const [ticker, setTicker] = useState("");
  const [days, setDays] = useState(10);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState("");

  const fetchPredictions = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, days }),
      });

      const data = await response.json();
      if (data.error) setError(data.error);
      else setPredictions(data.future_prediction);
    } catch (err) {
      setError("Failed to fetch predictions");
    }
  };

  return (
    <div>
      <input value={ticker} onChange={(e) => setTicker(e.target.value)} />
      <input type="number" value={days} onChange={(e) => setDays(e.target.value)} />
      <button onClick={fetchPredictions}>Predict</button>
      {predictions.map((p) => (
        <p key={p.date}>{p.date}: ${p.prediction}</p>
      ))}
    </div>
  );
};

export default App;
