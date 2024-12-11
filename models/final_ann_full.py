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
import stock_pred_rnn_full as sprf

# Paths for input files
companies_input_path = '../data/companies.csv'
stock_data_path = '../data/latest/stock_data.parquet'
finance_news_path = '../data/finance_news.parquet'
sentiment_json_path = 'final_artifacts/sentiment_analysis_full.json'
stock_pred_json_path = 'final_artifacts/stock_pred_rnn_full.json'
model_path = 'ann_parameters'
output_path = 'final_output.json'
patience = 10  # early stopping patience

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StockPredictionANN(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 24,
        hidden_layers: list = [64, 32],
        learning_rate: float = 0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Dynamic layer creation
        layers = []
        prev_size = input_size
        for h_size in hidden_layers:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        
        # Final output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
    
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
        batch_size: int = 32
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        os.makedirs(model_path, exist_ok=True)
        
    def load_input_data(self, sample_size):
        """Load sentiment and price data from JSON files"""
        with open(sentiment_json_path, 'r') as f:
            sentiment_data = json.load(f)
        
        # load price data from parquet
        df = pd.read_parquet(stock_data_path)
        price_data = df.groupby('Symbol')['Close'].apply(np.array).to_dict()

        with open(companies_input_path, 'r') as file:
            first_line = file.readline().strip()
            tickers = [ticker.strip() for ticker in first_line.split(',')]
        
        # get price data for the last sample_size days
        price_data = {ticker: price_data[ticker][-sample_size:] for ticker in tickers if ticker in price_data}

        
        # Check if any tickers are missing
        missing_tickers = [ticker for ticker in tickers if ticker not in price_data]
        
        # remove tickers from sentiment_data that are not in price_data
        sentiment_data = {ticker: sentiment_data[ticker] for ticker in tickers if ticker in price_data}

        return sentiment_data, price_data
        
    
    def prepare_data(self, sentiment_data, rnn_predicted_data, price_data):
        """Prepare training data"""
        X, Y = [], []

        # sort data by ticker symbols to ensure alignment
        for sentiment in sentiment_data:
            for ticker in sentiment.keys():
                X.append(float(format(sentiment[ticker][0], '.3f')))
                X.append(float(format(rnn_predicted_data[ticker][i], '.3f')))
                Y.append(float(format(price_data[ticker][i], '.3f')))
        
        print(X)
        print(Y)
        exit()
        
        return np.array(X), np.array(Y)
    
    def setup_dataloaders(self, X, Y):
        """Create train and validation dataloaders"""
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader
    
    def train_model(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        max_epochs: int = 500
    ) -> Tuple[pl.LightningModule, Optional[str]]:
        """Train the ANN model"""
        if len(X) == 0:
            return None, "No valid data for training"
        
        # Setup data loaders
        train_loader, val_loader = self.setup_dataloaders(X, Y)
        
        # Initialize model
        model = StockPredictionANN(input_size = len(X), output_size = len(X))
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_path,
            filename='ann_model-{epoch}-{val_loss:.2f}',
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
            name='stock_ann'
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
    
    def predict(self, model, X_pred):
        """Make predictions"""
        model.eval()
        with torch.no_grad():
            X_pred_tensor = torch.FloatTensor(X_pred)
            predictions = model(X_pred_tensor)
            return predictions.numpy().flatten()

def main(train_again=True, save_predictions=True, sample_size=5):
    # Initialize pipeline
    pipeline = StockPredictionPipeline()

    # Load static input data
    sentiment_data, actual_price_data = pipeline.load_input_data(sample_size)
    rnn_predicted_data = []

    # Train model for sample_size times
    for i in range(sample_size):
        # Load dynamic input data
        raw_pred_data, _ = sprf.main(train_again=False, save_predictions=False, 
                                            loop_back=5, index_start_from_back=sample_size - i)
        rnn_predicted_data.append(raw_pred_data)
    
    # Duplicate sentiment data for sample_size times
    sentiment_data = {ticker: [sentiment_data[ticker]] * sample_size for ticker in sentiment_data}

    # TODO HERE
    # Prepare training data
    # Get column ith from actual_price_data
    X, Y = pipeline.prepare_data(sentiment_data, rnn_predicted_data, actual_price_data)

    # Train model
    model, error = pipeline.train_model(X, Y[0])
    
    if error:
        print(f"Training error: {error}")
        return None, error
    
    # Prepare prediction input
    X_pred = np.array([
        [sentiment_data.get(symbol, 0), price_data.get(symbol, 0)] 
        for symbol in sentiment_data.keys()
    ])
    
    # Make predictions
    predictions = pipeline.predict(model, X_pred)
    
    # Organize predictions
    prediction_dict = {
        symbol: round(pred, 2) 
        for symbol, pred in zip(sentiment_data.keys(), predictions)
    }
    
    # Save or print predictions
    if save_predictions:
        with open(output_path, "w") as f:
            json.dump(prediction_dict, f)
    else:
        print("\nPredictions:")
        for symbol, price in prediction_dict.items():
            print(f"{symbol}: Predicted price: ${price:.2f}")
    
    return prediction_dict, None

if __name__ == "__main__":
    predictions, errors = main()