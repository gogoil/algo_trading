import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def download_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

def calculate_moving_averages(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    return data

def generate_signals(data):
    data['Signal'] = 0.0
    data['Signal'][50:] = np.where(data['SMA_50'][50:] > data['SMA_200'][50:], 1.0, 0.0)
    data['Position'] = data['Signal'].diff()
    return data




class TradingBot:
    def __init__(self, strategy, data):
        self.strategy = strategy(data)
        self.data = data

    def backtest_strategy(self):
        self.strategy.generate_signals()
        self.data['Market Return'] = self.data['Close'].pct_change()
        self.data['Strategy Return'] = self.data['Market Return'] * self.data['Position'].shift(1)
        self.data['Cumulative Market Return'] = (1 + self.data['Market Return']).cumprod()
        self.data['Cumulative Strategy Return'] = (1 + self.data['Strategy Return']).cumprod()
        return self.data

    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.data['Cumulative Market Return'], label='Market Return')
        plt.plot(self.data['Cumulative Strategy Return'], label='Strategy Return')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    data = download_data(ticker, start_date, end_date)
    data = calculate_moving_averages(data)
    data = generate_signals(data)
    data = backtest_strategy(data)
    plot_results(data)
