import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Load financial data with Yahoo Finance
def load_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

def prepare_data(df, sequence_length=256):
    # Use 'Close' prices and 'Volume'
    close_data = df['Close'].values
    volume_data = df['Volume'].values
    
    # Normalize the data
    close_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()
    
    close_data = close_scaler.fit_transform(close_data.reshape(-1, 1))
    volume_data = volume_scaler.fit_transform(volume_data.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(close_data) - sequence_length):
        X.append(np.hstack((close_data[i:i+sequence_length], volume_data[i:i+sequence_length])))
        y.append(close_data[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, (close_scaler, volume_scaler)


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StockDataModule(pl.LightningDataModule):
    def __init__(self, df, batch_size=32):
        super().__init__()
        X, y, scaler = prepare_data(df)

        X_train, X_temp, y_train, y_temp = TimeSeriesSplit(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = TimeSeriesSplit(X_temp, y_temp, test_size=0.5, random_state=42)

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = StockDataset(self.X_train, self.y_train)
        self.val_dataset = StockDataset(self.X_val, self.y_val)
        self.test_dataset = StockDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,  batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

# Create the DataModule

if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    df = load_stock_data(ticker, start_date, end_date)


    data_module = StockDataModule(df)
    data_module.prepare_data()
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break