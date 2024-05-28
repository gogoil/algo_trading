class Strategy:
    def __init__(self, data):
        self.data = data

    def generate_signals(self):
        raise NotImplementedError("Subclasses must implement generate_signals method.")
