#!/usr/bin/env python3
"""
Train LSTM on Full Dataset

Trains LSTM model on full training dataset (517 sequences).
Can be run in background with: nohup python3 train_lstm_full.py > train_full.log 2>&1 &

Default hyperparameters:
- Lookback: 20
- Hidden size: 128
- Num layers: 2
- Dropout: 0.2
- Learning rate: 0.001
- Epochs: 50 (with early stopping)
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

# Add competition package to path
sys.path.append('../competition_package')
from utils import DataPoint, ScorerStepByStep

# Configuration
CONFIG = {
    'lookback': 20,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 64,
    'lr': 0.001,
    'epochs': 50,
    'early_stopping_patience': 10,
    'device': 'cpu',
    'random_seed': 42
}

# Set random seeds
np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])


class SequenceDataset(Dataset):
    """Creates sequences for LSTM training."""
    def __init__(self, df, lookback, feature_cols):
        self.sequences = []
        self.targets = []

        print(f"Creating dataset with lookback={lookback}...")
        for seq_ix in df['seq_ix'].unique():
            seq_data = df[df['seq_ix'] == seq_ix].sort_values('step_in_seq')
            features = seq_data[feature_cols].values

            for i in range(lookback, len(features)):
                self.sequences.append(features[i-lookback:i])
                self.targets.append(features[i])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        print(f"Created {len(self.sequences):,} examples")

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


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            predictions = model(sequences)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    print("="*70)
    print("LSTM TRAINING ON FULL DATASET")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Print configuration
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()

    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}\n")

    # Load full training data
    print("Loading data...")
    train_df = pd.read_parquet('../competition_package/datasets/train.parquet')

    # Split into train/val (80/20)
    all_seqs = train_df['seq_ix'].unique()
    n_train = int(0.8 * len(all_seqs))
    train_seqs = all_seqs[:n_train]
    val_seqs = all_seqs[n_train:]

    train_data = train_df[train_df['seq_ix'].isin(train_seqs)]
    val_data = train_df[train_df['seq_ix'].isin(val_seqs)]

    feature_cols = [col for col in train_df.columns
                   if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]
    n_features = len(feature_cols)

    print(f"Train: {len(train_data):,} rows, {len(train_seqs)} sequences")
    print(f"Val:   {len(val_data):,} rows, {len(val_seqs)} sequences")
    print(f"Features: {n_features}\n")

    # Create datasets
    train_dataset = SequenceDataset(train_data, CONFIG['lookback'], feature_cols)
    val_dataset = SequenceDataset(val_data, CONFIG['lookback'], feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")

    # Create model
    print("Creating model...")
    model = LSTMPredictor(
        input_size=n_features,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}\n")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Create models directory
    Path('models').mkdir(exist_ok=True)

    # Training loop
    print(f"Training for up to {CONFIG['epochs']} epochs...\n")
    start_time = time.time()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(CONFIG['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Scheduler step
        scheduler.step(val_loss)

        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {current_lr:.6f}", end="")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG,
                'feature_cols': feature_cols
            }
            torch.save(checkpoint, 'models/lstm_full_best.pt')
            torch.save(model.state_dict(), 'models/lstm_full_state.pt')

            print(" ✓ New best!")
            patience_counter = 0
        else:
            patience_counter += 1
            print()

        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('models/lstm_full_history.csv', index=False)
    print(f"\nHistory saved: models/lstm_full_history.csv")

    # Save final config and results
    results = {
        'config': CONFIG,
        'best_val_loss': best_val_loss,
        'total_epochs': epoch + 1,
        'training_time_minutes': elapsed / 60,
        'n_parameters': n_params,
        'n_train_examples': len(train_dataset),
        'n_val_examples': len(val_dataset),
        'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('models/lstm_full_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: models/lstm_full_results.json")
    print(f"Model saved: models/lstm_full_best.pt")
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
