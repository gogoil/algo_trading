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
