#!/usr/bin/env python3
"""
Train LSTM with R² Metric

Uses R² for early stopping and learning rate scheduling instead of MSE.
Includes proper logging with unbuffered output.

Test run: python3 train_lstm_r2.py --test
Full run: python3 train_lstm_r2.py
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import time
import json
import argparse
from datetime import datetime
from sklearn.metrics import r2_score

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Add competition package to path
sys.path.append('../competition_package')


class SequenceDataset(Dataset):
    """Creates sequences for LSTM training."""
    def __init__(self, df, lookback, feature_cols):
        self.sequences = []
        self.targets = []

        print(f"Creating dataset with lookback={lookback}...", flush=True)
        for seq_ix in df['seq_ix'].unique():
            seq_data = df[df['seq_ix'] == seq_ix].sort_values('step_in_seq')
            features = seq_data[feature_cols].values

            for i in range(lookback, len(features)):
                self.sequences.append(features[i-lookback:i])
                self.targets.append(features[i])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        print(f"Created {len(self.sequences):,} examples", flush=True)

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


def calculate_r2(predictions, targets):
    """Calculate R² score for all features."""
    # Convert to numpy
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()

    # Calculate R² for each feature
    r2_scores = []
    for i in range(predictions.shape[1]):
        r2 = r2_score(targets[:, i], predictions[:, i])
        r2_scores.append(r2)

    return np.mean(r2_scores)


def train_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    device_type = str(device).split(':')[0]  # 'mps', 'cuda', or 'cpu'

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if use_amp and device_type in ['mps', 'cuda']:
            # Mixed precision training
            with autocast(device_type=device_type):
                predictions = model(sequences)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        all_preds.append(predictions.detach())
        all_targets.append(targets.detach())

    avg_loss = total_loss / len(loader)

    # Calculate R²
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = calculate_r2(all_preds, all_targets)

    return avg_loss, r2


def validate(model, loader, criterion, device, use_amp=False):
    """Validate model with optional mixed precision."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    device_type = str(device).split(':')[0]  # 'mps', 'cuda', or 'cpu'

    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            if use_amp and device_type in ['mps', 'cuda']:
                # Mixed precision inference
                with autocast(device_type=device_type):
                    predictions = model(sequences)
                    loss = criterion(predictions, targets)
            else:
                # Standard FP32 inference
                predictions = model(sequences)
                loss = criterion(predictions, targets)

            total_loss += loss.item()
            all_preds.append(predictions)
            all_targets.append(targets)

    avg_loss = total_loss / len(loader)

    # Calculate R²
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = calculate_r2(all_preds, all_targets)

    return avg_loss, r2


def detect_device(device_arg):
    """Detect and return the best available device."""
    print("\n" + "="*70, flush=True)
    print("DEVICE DETECTION", flush=True)
    print("="*70, flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"Requested device: {device_arg}", flush=True)
    print(flush=True)

    # Check available backends
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    has_cuda = torch.cuda.is_available()

    print("Available backends:", flush=True)
    print(f"  CPU: ✓ Always available", flush=True)
    print(f"  MPS (Apple Silicon): {'✓ Available' if has_mps else '✗ Not available'}", flush=True)
    print(f"  CUDA (NVIDIA): {'✓ Available' if has_cuda else '✗ Not available'}", flush=True)
    print(flush=True)

    # Select device
    if device_arg == 'auto':
        if has_mps:
            device = 'mps'
            print("Auto-selected: MPS (Apple Metal GPU)", flush=True)
        elif has_cuda:
            device = 'cuda'
            print("Auto-selected: CUDA (NVIDIA GPU)", flush=True)
        else:
            device = 'cpu'
            print("Auto-selected: CPU (no GPU available)", flush=True)
    elif device_arg == 'mps':
        if has_mps:
            device = 'mps'
            print("Using: MPS (Apple Metal GPU)", flush=True)
        else:
            print("WARNING: MPS requested but not available. Falling back to CPU.", flush=True)
            device = 'cpu'
    elif device_arg == 'cuda':
        if has_cuda:
            device = 'cuda'
            print("Using: CUDA (NVIDIA GPU)", flush=True)
        else:
            print("WARNING: CUDA requested but not available. Falling back to CPU.", flush=True)
            device = 'cpu'
    else:
        device = 'cpu'
        print("Using: CPU", flush=True)

    print(f"\nFinal device: {device}", flush=True)
    return device, has_mps, has_cuda


def main(test_mode=False, device_arg='auto', use_mixed_precision=False):
    # Create timestamp for versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*70, flush=True)
    print("LSTM TRAINING WITH R² METRIC", flush=True)
    print("="*70, flush=True)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Mode: {'TEST (tiny dataset)' if test_mode else 'FULL'}", flush=True)
    print(f"Mixed Precision: {'ENABLED (FP16)' if use_mixed_precision else 'DISABLED (FP32)'}", flush=True)
    print(f"Version: {timestamp}", flush=True)
    print(flush=True)

    # Detect device
    device_str, has_mps, has_cuda = detect_device(device_arg)

    # Configuration
    CONFIG = {
        'lookback': 20,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32 if test_mode else 64,
        'lr': 0.001,
        'epochs': 10 if test_mode else 50,
        'early_stopping_patience': 3 if test_mode else 10,
        'lr_scheduler_patience': 2 if test_mode else 5,
        'device': device_str,
        'device_requested': device_arg,
        'has_mps': has_mps,
        'has_cuda': has_cuda,
        'mixed_precision': use_mixed_precision,
        'pytorch_version': torch.__version__,
        'random_seed': 42,
        'test_mode': test_mode,
        'timestamp': timestamp
    }

    # Print configuration
    print("Configuration:", flush=True)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}", flush=True)
    print(flush=True)

    # Set random seeds
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])

    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}", flush=True)
    print(flush=True)

    # Load data
    print("Loading data...", flush=True)
    if test_mode:
        train_df = pd.read_parquet('data/tiny_train.parquet')
        val_df = pd.read_parquet('data/tiny_val.parquet')
        model_prefix = f'lstm_tiny_r2_{timestamp}'
    else:
        full_df = pd.read_parquet('../competition_package/datasets/train.parquet')
        # Split into train/val (80/20)
        all_seqs = full_df['seq_ix'].unique()
        n_train = int(0.8 * len(all_seqs))
        train_seqs = all_seqs[:n_train]
        val_seqs = all_seqs[n_train:]
        train_df = full_df[full_df['seq_ix'].isin(train_seqs)]
        val_df = full_df[full_df['seq_ix'].isin(val_seqs)]
        model_prefix = f'lstm_full_r2_{timestamp}'

    feature_cols = [col for col in train_df.columns
                   if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]
    n_features = len(feature_cols)

    print(f"Train: {len(train_df):,} rows, {train_df['seq_ix'].nunique()} sequences", flush=True)
    print(f"Val:   {len(val_df):,} rows, {val_df['seq_ix'].nunique()} sequences", flush=True)
    print(f"Features: {n_features}", flush=True)
    print(flush=True)

    # Create datasets
    train_dataset = SequenceDataset(train_df, CONFIG['lookback'], feature_cols)
    val_dataset = SequenceDataset(val_df, CONFIG['lookback'], feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"Train batches: {len(train_loader)}", flush=True)
    print(f"Val batches: {len(val_loader)}", flush=True)
    print(flush=True)

    # Create model
    print("Creating model...", flush=True)
    model = LSTMPredictor(
        input_size=n_features,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}", flush=True)
    print(flush=True)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # IMPORTANT: Use 'max' mode for R² (higher is better)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=CONFIG['lr_scheduler_patience']
    )

    # Mixed precision scaler
    scaler = None
    if CONFIG['mixed_precision']:
        device_type = str(device).split(':')[0]
        if device_type in ['mps', 'cuda']:
            scaler = GradScaler(device_type)
            print(f"Mixed precision enabled with GradScaler for {device_type}", flush=True)
        else:
            print("WARNING: Mixed precision requested but device is CPU. Using FP32.", flush=True)
            CONFIG['mixed_precision'] = False
    print(flush=True)

    # Create models directory
    Path('models').mkdir(exist_ok=True)

    # Training loop
    print(f"Training for up to {CONFIG['epochs']} epochs...", flush=True)
    print("Note: Early stopping and LR scheduling based on val R²", flush=True)
    print(flush=True)

    start_time = time.time()

    best_val_r2 = -float('inf')  # Best R² (higher is better)
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': [],
        'lr': []
    }

    for epoch in range(CONFIG['epochs']):
        # Train
        train_loss, train_r2 = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=CONFIG['mixed_precision']
        )

        # Validate
        val_loss, val_r2 = validate(
            model, val_loader, criterion, device,
            use_amp=CONFIG['mixed_precision']
        )

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Scheduler step (based on R², higher is better)
        scheduler.step(val_r2)

        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
              f"Loss: T={train_loss:.6f} V={val_loss:.6f} | "
              f"R²: T={train_r2:.6f} V={val_r2:.6f} | "
              f"LR={current_lr:.6f}", end="", flush=True)

        # Save best model (based on R²)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2

            # Save model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
                'train_r2': train_r2,
                'config': CONFIG,
                'feature_cols': feature_cols
            }
            torch.save(checkpoint, f'models/{model_prefix}_best.pt')
            torch.save(model.state_dict(), f'models/{model_prefix}_state.pt')

            print(" ✓ New best!", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1
            print(flush=True)

        # Early stopping (based on R²)
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch+1} (no R² improvement for {patience_counter} epochs)", flush=True)
            break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes", flush=True)
    print(f"Best validation R²: {best_val_r2:.6f}", flush=True)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'models/{model_prefix}_history.csv', index=False)
    print(f"\nHistory saved: models/{model_prefix}_history.csv", flush=True)

    # Save final config and results
    results = {
        'config': CONFIG,
        'best_val_r2': best_val_r2,
        'best_val_loss': float(checkpoint['val_loss']),
        'total_epochs': epoch + 1,
        'training_time_minutes': elapsed / 60,
        'n_parameters': n_params,
        'n_train_examples': len(train_dataset),
        'n_val_examples': len(val_dataset),
        'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(f'models/{model_prefix}_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: models/{model_prefix}_results.json", flush=True)
    print(f"Model saved: models/{model_prefix}_best.pt", flush=True)
    print("\n" + "="*70, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("="*70, flush=True)
    print(f"Best R²: {best_val_r2:.6f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM with R² metric')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with tiny dataset')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'mps', 'cuda', 'auto'],
                       help='Device to use for training (default: auto-detect)')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision (FP16) training for faster performance on GPU')
    args = parser.parse_args()

    try:
        main(test_mode=args.test, device_arg=args.device,
             use_mixed_precision=args.mixed_precision)
    except Exception as e:
        print(f"\nERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
