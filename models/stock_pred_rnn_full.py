import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import json
from time import sleep


companies_input_path = '../data/companies.csv'
stock_data_path = '../data/latest/stock_data.parquet'
model_path = 'rnn_parameters'
output_path = 'final_artifacts/stock_pred_rnn_full.json'
patience = 10 # after this epochs, stop training if no improvement
# patience = 500 # almost working case

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StockPredictionModule(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        num_layers: int = 3,
        learning_rate: float = 0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.hparams.num_layers, batch_size, 
                        self.hparams.hidden_size, device=self.device)
        
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class StockPredictionPipeline:
    def __init__(
        self,
        model_path: str = model_path,
        index_start_from_back: int = 5,
        look_back: int = 15,
        batch_size: int = 32
    ):
        self.model_path = model_path
        self.look_back = look_back
        self.batch_size = batch_size
        self.index_start_from_back = index_start_from_back
        os.makedirs(model_path, exist_ok=True)
        
    def prepare_data(self, price_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sequences for training"""
        X, Y = [], []

        length = len(price_data) - self.index_start_from_back - self.look_back
        if length <= 0:
            print(f"Insufficient data for {symbol}. Need at least {self.look_back + 1} days.")
            length = 0
        for i in range(length):
            X.append(price_data[i:(i + self.look_back)])
            Y.append(price_data[i + self.look_back])
            
        return np.array(X).reshape(-1, self.look_back, 1), np.array(Y)
    
    def setup_dataloaders(self, X: np.ndarray, Y: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """Creates train and validation dataloaders"""
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader
    
    def train_model(
        self,
        symbol: str,
        price_data: np.ndarray,
        max_epochs: int = 5000
    ) -> Tuple[pl.LightningModule, Optional[str]]:
        """Trains or loads a model for the given symbol"""
        if len(price_data) <= self.look_back:
            return None, f"Insufficient data for {symbol}. Need at least {self.look_back + 1} days."
            
        # Prepare data
        X, Y = self.prepare_data(price_data)
        if len(X) == 0:
            return None, f"Could not create sequences for {symbol}"
            
        # Setup data loaders
        train_loader, val_loader = self.setup_dataloaders(X, Y)
        
        # Initialize model
        model = StockPredictionModule()
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.model_path, symbol),
            filename=f'{symbol}-{{epoch}}-{{val_loss:.2f}}',
            monitor='val_loss',
            mode='min'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min'
        )
        
        # Setup logger
        logger = TensorBoardLogger(
            save_dir='lightning_logs',
            name=symbol
        )
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger,
            accelerator='auto'
        )
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        return model, None
    
    def predict(self, model: pl.LightningModule, last_sequence: np.ndarray) -> float:
        """Makes prediction for the next day"""
        model.eval()
        with torch.no_grad():
            X_pred = torch.FloatTensor(last_sequence.reshape(1, self.look_back, 1))
            X_pred = X_pred.to(model.device)
            prediction = model(X_pred)
            return prediction.item()
    
    def load_model(self, symbol: str) -> Optional[pl.LightningModule]:
        """Loads a saved model if it exists"""
        model_dir = os.path.join(self.model_path, symbol)
        if os.path.exists(model_dir):
            print(f"Model directory: {model_dir}")
            checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
            if checkpoints:
                print(f"Found {len(checkpoints)} checkpoints for {symbol}")
                latest_checkpoint = sorted(checkpoints)[-1]
                model = StockPredictionModule.load_from_checkpoint(
                    os.path.join(model_dir, latest_checkpoint)
                )
                print(f"Loaded existing model for {symbol}")
                return model
        return None

def read_input_data(filename=companies_input_path):
    """
    Read the first line of the CSV file containing ticker symbols
    """
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        tickers = [ticker.strip() for ticker in first_line.split(',')]
    return tickers

def main(train_again=True, save_predictions=True, loop_back=10, index_start_from_back=5):
    # Initialize pipeline
    pipeline = StockPredictionPipeline(index_start_from_back=index_start_from_back, 
                                        look_back=loop_back)
    
    # Load your data
    df = pd.read_parquet(stock_data_path)

    # Load input
    input_companies = read_input_data(companies_input_path)
    
    # Organize data by symbol
    symbol_data = df.groupby('Symbol')['Close'].apply(np.array).to_dict()
    predictions = {}
    errors = {}
    
    for symbol, price_data in symbol_data.items():
        if symbol in input_companies:
            print(f"\nProcessing {symbol}...")

            # Try to load existing model first
            model = pipeline.load_model(symbol)
            
            # If no existing model, train a new one
            if model is None or train_again:
                print("Training new model...")
                sleep(5)
                model, error = pipeline.train_model(symbol, price_data)
                if error:
                    errors[symbol] = error
                    continue
            
            # Make prediction
            last_sequence = price_data[-pipeline.look_back:]
            prediction = pipeline.predict(model, last_sequence)
            predictions[symbol] = round(prediction, 2)
    
    # Print results
    if save_predictions:
        with open(output_path, "w") as f:
            json.dump(predictions, f)
    else:
        print("\nPredictions:")
        for symbol, price in predictions.items():
            print(f"{symbol}: Next day closing price prediction: ${price:.2f}")
    
    if errors:
        print("\nErrors:")
        for symbol, error in errors.items():
            print(f"{symbol}: {error}")
    
    return predictions, errors

if __name__ == "__main__":
    predictions, errors = main()