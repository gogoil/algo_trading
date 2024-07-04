import os
import json
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from data.data_loader import StockDataModule
from model.cnn_model import CnnModel

def load_training_params(file_path='model/training_params.json'):
    with open(file_path, 'r') as f:
        params = json.load(f)
    return params

def main():
    # Load training parameters
    params = load_training_params()

    # Create data module
    data_module = StockDataModule(
        data_dir=params['data_dir'],
        batch_size=params['batch_size'],
        num_workers=params['num_workers']
    )

    # Create model
    model = CnnModel(learning_rate=params['learning_rate'])

    # Set up logging
    current_time = datetime.now().strftime('%H_%M_%S__%d%m%Y')
    log_dir = os.path.join('trained_models', f'trained_on_{current_time}')
    wandb_logger = WandbLogger(log_model="all")

    wandb.init(
        # set the wandb project where this run will be logged
        project="Algo_Trading",
        name = 'low_lr_30_epoches',
        dir = log_dir,
        # track hyperparameters and run metadata
        config={
        "learning_rate": params['learning_rate'],
        "architecture": "CNN",
        "dataset": "yahoo_finance",
        "epochs": params['max_epochs'],
        }
    )

    # Set up checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename='epoch={epoch:03d}-loss={val_loss:.3f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=params['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        #gpus=1 if params['use_gpu'] else 0,
        devices="auto", accelerator="auto",
        log_every_n_steps=10,
        precision="16-mixed" if params['use_mixed_precision'] else "32-true"
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

    # Save the last checkpoint
    trainer.save_checkpoint(os.path.join(log_dir, 'last.ckpt'))

if __name__ == '__main__':
    main()