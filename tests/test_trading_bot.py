import pytest
import pandas as pd
from src.trading_bot import download_data, calculate_moving_averages, generate_signals, backtest_strategy

@pytest.fixture
def sample_data():
    dates = pd.date_range('2020-01-01', periods=200)
    data = pd.DataFrame(index=dates)
    data['Close'] = range(200)
    return data

def test_download_data():
    data = download_data('AAPL', '2020-01-01', '2023-01-01')
    assert not data.empty

def test_calculate_moving_averages(sample_data):
    data = calculate_moving_averages(sample_data)
    assert 'SMA_50' in data.columns
    assert 'SMA_200' in data.columns

def test_generate_signals(sample_data):
    sample_data['SMA_50'] = sample_data['Close'].rolling(window=50).mean()
    sample_data['SMA_200'] = sample_data['Close'].rolling(window=200).mean()
    data = generate_signals(sample_data)
    assert 'Signal' in data.columns
    assert 'Position' in data.columns

def test_backtest_strategy(sample_data):
    sample_data['SMA_50'] = sample_data['Close'].rolling(window=50).mean()
    sample_data['SMA_200'] = sample_data['Close'].rolling(window=200).mean()
    sample_data['Signal'] = 0.0
    sample_data['Signal'][50:] = 1.0
    sample_data['Position'] = sample_data['Signal'].diff()
    data = backtest_strategy(sample_data)
    assert 'Strategy Return' in data.columns
    assert 'Cumulative Market Return' in data.columns
    assert 'Cumulative Strategy Return' in data.columns
    assert 'Market Return' in data.columns
