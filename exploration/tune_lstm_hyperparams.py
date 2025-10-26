#!/usr/bin/env python3
"""
LSTM Hyperparameter Tuning

Grid search over key hyperparameters to find best LSTM configuration.
Can be run in background: nohup python3 tune_lstm_hyperparams.py > tune.log 2>&1 &

Hyperparameters to tune:
- Lookback: [10, 20, 30, 50]
- Hidden size: [64, 128, 256]
- Num layers: [1, 2]
- Dropout: [0.1, 0.2, 0.3]
- Learning rate: [0.0001, 0.001, 0.01]

Total combinations: Reduced grid search (~20-30 runs)
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import json
from datetime import datetime
from itertools import product

# Add competition package to path
sys.path.append('../competition_package')
from utils import DataPoint, ScorerStepByStep


class SequenceDataset(Dataset):
    """Creates sequences for LSTM training."""
    def __init__(self, df, lookback, feature_cols):
        self.sequences = []
        self.targets = []

        for seq_ix in df['seq_ix'].unique():
            seq_data = df[df['seq_ix'] == seq_ix].sort_values('step_in_seq')
            features = seq_data[feature_cols].values

            for i in range(lookback, len(features)):
                self.sequences.append(features[i-lookback:i])
                self.targets.append(features[i])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])


class LSTMPredictor(nn.Module):
    """LSTM-based sequence predictor."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def train_model(config, train_loader, val_loader, device, n_features):
    """Train a single model configuration."""
    # Create model
    model = LSTMPredictor(
        input_size=n_features,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5  # Reduced for tuning

    # Quick training (max 15 epochs for tuning)
    for epoch in range(15):
        # Train
        model.train()
        train_loss = 0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(seqs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                preds = model(seqs)
                val_loss += criterion(preds, targets).item()
        val_loss /= len(val_loader)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            break

    return best_val_loss, epoch + 1


def main():
    print("="*70)
    print("LSTM HYPERPARAMETER TUNING")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    device = torch.device('cpu')
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data (use sample for faster tuning)
    print("Loading sample data for tuning...")
    train_df = pd.read_parquet('data/sample_train.parquet')
    val_df = pd.read_parquet('data/sample_val.parquet')

    feature_cols = [col for col in train_df.columns
                   if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]
    n_features = len(feature_cols)

    print(f"Train: {len(train_df):,} rows, {train_df['seq_ix'].nunique()} sequences")
    print(f"Val:   {len(val_df):,} rows, {val_df['seq_ix'].nunique()} sequences")
    print(f"Features: {n_features}\n")

    # Define hyperparameter grid (reduced for speed)
    param_grid = {
        'lookback': [10, 20, 30],
        'hidden_size': [64, 128, 256],
        'num_layers': [1, 2],
        'dropout': [0.1, 0.2],
        'lr': [0.0005, 0.001, 0.002],
        'batch_size': [64]  # Fixed
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_configs = [dict(zip(keys, combo)) for combo in product(*values)]

    print(f"Total configurations to test: {len(all_configs)}\n")

    # Results storage
    results = []

    # Test each configuration
    for i, config in enumerate(all_configs):
        print(f"\n[{i+1}/{len(all_configs)}] Testing configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")

        try:
            # Create datasets for this lookback
            train_dataset = SequenceDataset(train_df, config['lookback'], feature_cols)
            val_dataset = SequenceDataset(val_df, config['lookback'], feature_cols)

            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

            # Train model
            start_time = time.time()
            val_loss, epochs = train_model(config, train_loader, val_loader, device, n_features)
            elapsed = time.time() - start_time

            # Save result
            result = {
                **config,
                'val_loss': val_loss,
                'epochs': epochs,
                'training_time': elapsed
            }
            results.append(result)

            print(f"  Val loss: {val_loss:.6f} | Epochs: {epochs} | Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_loss')

    # Save results
    Path('models').mkdir(exist_ok=True)
    results_df.to_csv('models/hyperparameter_tuning_results.csv', index=False)

    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)

    print("\nTop 5 configurations:")
    print(results_df.head(5).to_string(index=False))

    print(f"\nBest configuration:")
    best = results_df.iloc[0]
    for k in ['lookback', 'hidden_size', 'num_layers', 'dropout', 'lr']:
        print(f"  {k}: {best[k]}")
    print(f"  val_loss: {best['val_loss']:.6f}")

    # Save best config
    best_config = best.to_dict()
    with open('models/best_hyperparameters.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\nResults saved:")
    print(f"  models/hyperparameter_tuning_results.csv")
    print(f"  models/best_hyperparameters.json")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
