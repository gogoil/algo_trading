import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CnnModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, number_of_channels = 3):
        super().__init__()
        self.number_of_channels = number_of_channels
        self.save_hyperparameters()
        # Parallel 1D convolution blocks
        self.conv1 = nn.Conv1d(number_of_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(number_of_channels, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(number_of_channels, 64, kernel_size=7, stride=2, padding=3)
        
        # Additional layers
        self.pool = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(192, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.conv6 = nn.Conv1d(256, 16, kernel_size=7, padding=3, stride=2)

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1)

        self.dropout = nn.Dropout(0.5)


        # for inference:
        self.x_tensor = None
        self.y_tensor = None
        self.probabilities_tensor = None

    def forward(self, x):
        # x shape: [batch, 3, 32, 32]
        x = x.view(x.size(0), self.number_of_channels, -1)  # Flatten to [batch, number_of_channels, 1024]
        
        # Parallel convolutions
        x1 = F.relu(self.conv1(x)) # torch.jit.fork(self.conv1, x)
        x2 = F.relu(self.conv2(x)) # torch.jit.fork(self.conv1, x)
        x3 = F.relu(self.conv3(x))  # torch.jit.fork(self.conv1, x)
        #[batch, 64, 512]

        # out1 = torch.jit.wait(fut1)
        # out2 = torch.jit.wait(fut2)
        # out3 = torch.jit.wait(fut3)

        # Concatenate the outputs
        x = torch.cat((x1, x2, x3), dim=1)  # torch.Size([batch, 192, 512])
        x = self.pool(x) # torch.Size([batch, 128, 256])

        # Additional layers
        x = F.relu(self.conv4(x)) # torch.Size([batch, 128, 256])
        x = F.relu(self.conv5(x)) # torch.Size([batch, 128, 256])
        x= F.relu(self.conv6(x)) # torch.Size([batch, 64, 256])
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        prob = F.softmax(self(x),dim=1)
        if self.x_tensor is None:
            self.x_tensor = x
            self.y_tensor = y
            self.probabilities_tensor = prob
        else:
            self.x_tensor = torch.cat((self.x_tensor, x), dim=0)
            self.y_tensor = torch.cat((self.y_tensor, y), dim=0)
            self.probabilities_tensor = torch.cat((self.probabilities_tensor, prob), dim=0)
        return x, y, prob

    def on_prediction_end(self):
        print('prediction_ended')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def accuracy(preds, y):
    return torch.tensor(torch.sum(preds == y).item() / len(preds))

# Example usage
if __name__ == "__main__":
    model = CnnModel(number_of_channels=3)
    print(model)
    
    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Test forward pass
    x = torch.randn(30, 3, 1024)
    output = model(x)
    print(f"Output shape: {output.shape}")