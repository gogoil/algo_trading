import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def download_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

def calculate_moving_averages(data, short_window, long_window):
    data['SMA_50'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_200'] = data['Close'].rolling(window=long_window).mean()
    return data

def generate_signals(data):
    data['Signal'] = 0.0
    data['Signal'][50:] = np.where(data['SMA_50'][50:] > data['SMA_200'][50:], 1.0, 0.0)
    data['Position'] = data['Signal'].diff()
    return data

def backtest_strategy(data):
    data['Market Return'] = data['Close'].pct_change()
    data['Strategy Return'] = data['Market Return'] * data['Position'].shift(1)
    data['Cumulative Market Return'] = (1 + data['Market Return']).cumprod()
    data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()
    return data

def plot_results(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Cumulative Market Return'], label='Market Return')
    plt.plot(data['Cumulative Strategy Return'], label='Strategy Return')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    data = download_data(ticker, start_date, end_date)
    data = calculate_moving_averages(data, short_window=50, long_window=200)
    data = generate_signals(data)
    data = backtest_strategy(data)
    plot_results(data)