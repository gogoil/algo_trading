import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.strategy import *

def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    return yf.download(ticker, start=start_date, end=end_date)

def calculate_moving_averages(data: pd.DataFrame)-> pd.DataFrame:
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    return data

def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    data['Signal'] = 0.0
    data['Signal'][50:] = np.where(data['SMA_50'][50:] > data['SMA_200'][50:], 1.0, 0.0)
    data['Position'] = data['Signal'].diff()
    return data




class TradingBot:
    def __init__(self, strategy: Strategy, data: pd.DataFrame):
        self.strategy = strategy(data)
        self.data = data

    def backtest_strategy(self)-> pd.DataFrame:
        self.strategy.generate_signals()
        self.data['Market Return'] = self.data['Close'].pct_change()
        self.data['Strategy Return'] = self.data['Market Return'] * self.data['Position'].shift(1)
        self.data['Cumulative Market Return'] = (1 + self.data['Market Return']).cumprod()
        self.data['Cumulative Strategy Return'] = (1 + self.data['Strategy Return']).cumprod()
        return self.data

    def plot_results(self, stock:str):
        plt.figure(figsize=(10, 5))
        plt.plot(self.data['Cumulative Market Return'], label='Market Return')
        plt.plot(self.data['Cumulative Strategy Return'], label='Strategy Return')
        plt.title(f'{stock} - {str(self.strategy)}')
        plt.legend()
        plt.show()

    def run_backtest(self) -> None:
        for ticker, stock_data in self.data.items():
            strategy_instance = self.strategy(stock_data)
            strategy_instance.generate_signals()

if __name__ == "__main__":
    # tickers = ["TSLA", "BTC-USD", 'SPY', 'MSTR']
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    start_date = "2021-01-01"
    end_date = "2024-05-27"
    data = {}
    for ticker in tickers:
        data[ticker] = download_data(ticker, start_date, end_date)

    strategies = [AutoregressiveStrategy]  # Add any other strategies you want to test

    for ticker, stock_data in data.items():
        for strategy in strategies:
            bot = TradingBot(strategy, stock_data)
            backtest_results = bot.backtest_strategy()
            bot.plot_results(ticker)
