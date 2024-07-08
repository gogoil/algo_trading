import os
import torch
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

# Assuming you've already defined your StockPredictionModel class
from model.cnn_model import CnnModel

class StockPredictor:
    def __init__(self, model_path, sequence_length=256):
        self.sequence_length = sequence_length
        self.model = CnnModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.close_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()

    def load_and_prepare_data(self, ticker, start_date):
        # Load data
        stock = yf.Ticker(ticker)
        self.df = stock.history(start=start_date)

        # Prepare data
        close_data = self.df['Close'].values.reshape(-1, 1)
        volume_data = self.df['Volume'].values.reshape(-1, 1)

        # Fit scalers and transform data
        self.close_scaler.fit(close_data)
        self.volume_scaler.fit(volume_data)
        
        close_scaled = self.close_scaler.transform(close_data)
        volume_scaled = self.volume_scaler.transform(volume_data)

        # Create sequence
        sequence = np.hstack((close_scaled[-self.sequence_length:], volume_scaled[-self.sequence_length:]))
        return torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension

    def predict_next_n_steps(self, ticker, start_date, n_steps):
        sequence = self.load_and_prepare_data(ticker, start_date)
        predictions = []
        print(sequence)
        for _ in range(n_steps):
            # Get prediction
            with torch.no_grad():
                pred = self.model(sequence).item()
            predictions.append(pred)

            # Update sequence
            new_data_point = torch.FloatTensor([[pred, 0]])  # We don't have future volume, so use 0
            sequence = torch.cat((sequence[:, 1:, :], new_data_point.unsqueeze(1)), dim=1)

        # Inverse transform predictions
        predictions = self.close_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()

    def plot_predictions(self, predictions, n_steps):
        # Create future dates for predictions
        last_date = self.df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, n_steps + 1)]

        # Plot the historical data
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Close'], label='Historical Data')

        # Plot the predictions
        plt.plot(future_dates, predictions, label='Predictions', color='red')

        plt.title(f"{self.df.index[0].date()} to {future_dates[-1].date()} Stock Price")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Usage
if __name__ == "__main__":
    model_path = "trained_models/trained_on_09_56_43__07072024/last.ckpt"
    predictor = StockPredictor(model_path)

    ticker = "meta"
    start_date = "2023-01-01"
    n_steps = 30

    predictions = predictor.predict_next_n_steps(ticker, start_date, n_steps)
    
    print(f"Predictions for the next {n_steps} days for {ticker}:")
    for i, pred in enumerate(predictions, 1):
        print(f"Day {i}: ${pred:.2f}")

    # Plot the predictions
    predictor.plot_predictions(predictions, n_steps)