import React, { useState } from "react";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Container, Form, Button } from "react-bootstrap";

const StockPredictor = () => {
    const [ticker, setTicker] = useState("");
    const [predictions, setPredictions] = useState([]);

    const fetchPredictions = async () => {
        try {
            const response = await axios.post("http://127.0.0.1:5000/predict", { ticker });
            setPredictions(response.data.predictions.map((price, index) => ({ day: index + 1, price: price[0] })));
        } catch (error) {
            console.error("Error fetching predictions:", error);
        }
    };

    return (
        <Container className="mt-5 text-center">
            <h2>📈 Stock Price Prediction</h2>
            <Form>
                <Form.Group>
                    <Form.Control
                        type="text"
                        placeholder="Enter stock ticker (e.g., AAPL)"
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                    />
                </Form.Group>
                <Button variant="primary" className="mt-3" onClick={fetchPredictions}>
                    Predict
                </Button>
            </Form>

            {predictions.length > 0 && (
                <ResponsiveContainer width="100%" height={300} className="mt-4">
                    <LineChart data={predictions}>
                        <XAxis dataKey="day" label={{ value: "Days", position: "insideBottom", offset: -5 }} />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="price" stroke="#8884d8" strokeWidth={2} />
                    </LineChart>
                </ResponsiveContainer>
            )}
        </Container>
    );
};

export default StockPredictor;
