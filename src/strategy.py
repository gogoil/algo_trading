class Strategy:
    def __init__(self, data):
        self.data = data

    def generate_signals(self):
        raise NotImplementedError("Subclasses must implement generate_signals method.")

class MovingAverageCrossover(Strategy):
    def __init__(self,data):
        super().__init__(data)


    def generate_signals(self):
        # Calculate moving averages
        self.data['Short_MA'] = self.data['Close'].rolling(window=50).mean()
        self.data['Long_MA'] = self.data['Close'].rolling(window=200).mean()

        # Generate signals based on crossovers
        self.data['Position'] = 0
        self.data.loc[self.data['Short_MA'] > self.data['Long_MA'], 'Position'] = 1
        self.data.loc[self.data['Short_MA'] < self.data['Long_MA'], 'Position'] = -1

    def __str__(self):
        return "Moving Average Crossover Strategy"

class RSIStrategy(Strategy):
    def __init__(self, data):
        super().__init__(data)
    
    def generate_signals(self):
        # Calculate RSI
        self.data['delta'] = self.data['Close'].diff()
        self.data['gain'] = self.data['delta'].where(self.data['delta'] > 0, 0)
        self.data['loss'] = -self.data['delta'].where(self.data['delta'] < 0, 0)
        self.data['avg_gain'] = self.data['gain'].rolling(window=14).mean()
        self.data['avg_loss'] = self.data['loss'].rolling(window=14).mean()
        self.data['rs'] = self.data['avg_gain'] / self.data['avg_loss']
        self.data['rsi'] = 100 - (100 / (1 + self.data['rs']))
        
        # Generate signals based on RSI levels
        self.data['Position'] = 0
        self.data.loc[self.data['rsi'] > 70, 'Position'] = -1
        self.data.loc[self.data['rsi'] < 30, 'Position'] = 1

    def __str__(self):
        return "RSI Strategy"

class AutoregressiveStrategy(Strategy):
    def __init__(self, data, lag):
        super().__init__(data)
        self.lag = lag
        self.beta = np.zeros(lag)
    
    def fit(self, data):
        # Fit autoregressive model
        for i in range(self.lag):
            self.beta[i] = np.corrcoef(data['Close'][i:], data['Close'][:-i])[0, 1]

    def generate_signals(self):
        # Calculate autoregressive values
        pass
    def __str__(self):
        return "Autoregressive Strategy"