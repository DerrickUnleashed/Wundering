#!/usr/bin/env python3
"""
Broad Hyperparameter Search - Phase 1

Random search over 17 hyperparameters to identify promising regions.
Uses 50% of data for fast iteration.

Usage:
  Test mode (tiny data, 5 configs): python3 tune_lstm_broad.py --test
  Full mode (50% data, 100 configs): python3 tune_lstm_broad.py
  Custom: python3 tune_lstm_broad.py --n_configs 50 --sample_pct 0.3
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
import argparse
from datetime import datetime
from sklearn.metrics import r2_score
import random
from tqdm import tqdm

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Add current directory to path
sys.path.append('.')
sys.path.append('../competition_package')

from models import create_model_from_config


class SequenceDataset(Dataset):
    """Creates sequences for RNN training."""
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


def calculate_r2(predictions, targets):
    """Calculate R² score for all features."""
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()

    r2_scores = []
    for i in range(predictions.shape[1]):
        r2 = r2_score(targets[:, i], predictions[:, i])
        r2_scores.append(r2)

    return np.mean(r2_scores)


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, targets)
        loss.backward()

        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        all_preds.append(predictions.detach())
        all_targets.append(targets.detach())

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = calculate_r2(all_preds, all_targets)

    return avg_loss, r2


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            predictions = model(sequences)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

            all_preds.append(predictions)
            all_targets.append(targets)

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = calculate_r2(all_preds, all_targets)

    return avg_loss, r2


def sample_hyperparameters():
    """Sample random hyperparameters from search space."""
    config = {
        # Architecture
        'lookback': random.choice([5, 10, 15, 20, 30, 50]),
        'hidden_size': random.choice([32, 64, 96, 128, 192, 256]),
        'num_layers': random.choice([1, 2, 3, 4]),
        'dropout': random.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        'bidirectional': random.choice([False, True]),
        'use_gru': random.choice([False, True]),

        # Output head
        'fc_num_layers': random.choice([1, 2, 3]),
        'fc_hidden_dims': random.choice([64, 128, 256]),
        'fc_activation': random.choice(['relu', 'tanh', 'gelu']),
        'fc_dropout': random.choice([0.0, 0.2, 0.3, 0.4]),
        'use_batch_norm': random.choice([False, True]),

        # Training
        'batch_size': random.choice([32, 64, 128, 256]),
        'lr': random.choice([0.0001, 0.0005, 0.001, 0.002, 0.005]),
        'weight_decay': random.choice([0, 0.0001, 0.001, 0.01]),
        'grad_clip': random.choice([None, 0.5, 1.0, 5.0]),
        'optimizer': random.choice(['adam', 'adamw']),
        'lr_patience': random.choice([3, 5, 7, 10])
    }

    return config


def detect_device(device_arg):
    """Detect and return the best available device."""
    # Check available backends
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    has_cuda = torch.cuda.is_available()

    # Select device
    if device_arg == 'auto':
        if has_mps:
            device = 'mps'
        elif has_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
    elif device_arg == 'mps':
        device = 'mps' if has_mps else 'cpu'
    elif device_arg == 'cuda':
        device = 'cuda' if has_cuda else 'cpu'
    else:
        device = 'cpu'

    return device, has_mps, has_cuda


def train_model(config, train_df, val_df, feature_cols, device, max_epochs=20, patience=5):
    """Train a single model configuration."""
    n_features = len(feature_cols)

    # Create datasets
    train_dataset = SequenceDataset(train_df, config['lookback'], feature_cols)
    val_dataset = SequenceDataset(val_df, config['lookback'], feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Create model
    model = create_model_from_config(config, n_features).to(device)

    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:  # adamw
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=config['lr_patience']
    )

    criterion = nn.MSELoss()

    # Training loop
    best_val_r2 = -float('inf')
    patience_counter = 0
    best_epoch = 0
    history = []

    for epoch in range(max_epochs):
        train_loss, train_r2 = train_epoch(
            model, train_loader, criterion, optimizer, device, config['grad_clip']
        )
        val_loss, val_r2 = validate(model, val_loader, criterion, device)

        scheduler.step(val_r2)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_r2': train_r2,
            'val_loss': val_loss,
            'val_r2': val_r2,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Print epoch progress
        marker = "✓" if val_r2 > best_val_r2 else " "
        gap = train_r2 - val_r2
        print(f"    Epoch {epoch+1:2d}/{max_epochs}: "
              f"Val R²={val_r2:.4f} Train R²={train_r2:.4f} Gap={gap:.4f} {marker}",
              flush=True)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopped at epoch {epoch+1}", flush=True)
            break

    # Calculate overfitting gap (train - val R² at best epoch)
    best_epoch_data = history[best_epoch - 1]
    overfitting_gap = best_epoch_data['train_r2'] - best_epoch_data['val_r2']

    return {
        'best_val_r2': best_val_r2,
        'best_train_r2': best_epoch_data['train_r2'],
        'overfitting_gap': overfitting_gap,
        'best_epoch': best_epoch,
        'total_epochs': len(history),
        'n_parameters': model.count_parameters(),
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(description='Broad hyperparameter search')
    parser.add_argument('--test', action='store_true', help='Test mode (tiny data, 5 configs)')
    parser.add_argument('--n_configs', type=int, default=100, help='Number of configurations to test')
    parser.add_argument('--sample_pct', type=float, default=0.5, help='Percentage of data to use')
    parser.add_argument('--max_epochs', type=int, default=20, help='Max epochs per config')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'mps', 'cuda', 'auto'],
                       help='Device to use for training (default: auto-detect)')
    args = parser.parse_args()

    print("="*70, flush=True)
    print("BROAD HYPERPARAMETER SEARCH - PHASE 1", flush=True)
    print("="*70, flush=True)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Mode: {'TEST' if args.test else 'FULL'}", flush=True)
    print(flush=True)

    # Detect device
    device_str, has_mps, has_cuda = detect_device(args.device)
    device = torch.device(device_str)

    print("Device Detection:", flush=True)
    print(f"  PyTorch version: {torch.__version__}", flush=True)
    print(f"  Requested: {args.device}", flush=True)
    print(f"  MPS available: {'✓' if has_mps else '✗'}", flush=True)
    print(f"  CUDA available: {'✓' if has_cuda else '✗'}", flush=True)
    print(f"  Using: {device_str.upper()}", flush=True)
    print(flush=True)

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("Loading data...", flush=True)
    if args.test:
        train_df = pd.read_parquet('data/tiny_train.parquet')
        val_df = pd.read_parquet('data/tiny_val.parquet')
        n_configs = 5
        max_epochs = 3
        output_file = 'models/test_broad_search_results.csv'
    else:
        full_df = pd.read_parquet('../competition_package/datasets/train.parquet')
        all_seqs = full_df['seq_ix'].unique()

        # Sample sequences
        n_sample = int(len(all_seqs) * args.sample_pct)
        sampled_seqs = np.random.choice(all_seqs, n_sample, replace=False)
        sampled_df = full_df[full_df['seq_ix'].isin(sampled_seqs)]

        # Split sampled data 80/20
        n_train = int(0.8 * len(sampled_seqs))
        train_seqs = sampled_seqs[:n_train]
        val_seqs = sampled_seqs[n_train:]

        train_df = sampled_df[sampled_df['seq_ix'].isin(train_seqs)]
        val_df = sampled_df[sampled_df['seq_ix'].isin(val_seqs)]

        n_configs = args.n_configs
        max_epochs = args.max_epochs
        output_file = f'models/broad_search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    feature_cols = [col for col in train_df.columns
                   if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]

    print(f"Train: {len(train_df):,} rows, {train_df['seq_ix'].nunique()} sequences", flush=True)
    print(f"Val:   {len(val_df):,} rows, {val_df['seq_ix'].nunique()} sequences", flush=True)
    print(f"Features: {len(feature_cols)}", flush=True)
    print(f"Configurations to test: {n_configs}", flush=True)
    print(f"Max epochs per config: {max_epochs}", flush=True)
    print(flush=True)

    # Create models directory
    Path('models').mkdir(exist_ok=True)

    # Run search
    results = []
    start_time = time.time()

    print("Starting hyperparameter search...", flush=True)
    print(flush=True)

    for i in tqdm(range(n_configs), desc="Searching"):
        config_start = time.time()

        # Sample hyperparameters
        config = sample_hyperparameters()

        try:
            # Print config details before training
            print(f"\n{'='*70}", flush=True)
            print(f"Config {i+1}/{n_configs} - Starting Training:", flush=True)
            print(f"  Architecture:", flush=True)
            print(f"    Lookback: {config['lookback']}, Hidden: {config['hidden_size']}, "
                  f"Layers: {config['num_layers']}, Dropout: {config['dropout']}", flush=True)
            print(f"    Bidirectional: {config['bidirectional']}, GRU: {config['use_gru']}", flush=True)
            print(f"  Output Head:", flush=True)
            print(f"    FC Layers: {config['fc_num_layers']}, FC Hidden: {config['fc_hidden_dims']}, "
                  f"Activation: {config['fc_activation']}", flush=True)
            print(f"    FC Dropout: {config['fc_dropout']}, Batch Norm: {config['use_batch_norm']}", flush=True)
            print(f"  Training:", flush=True)
            print(f"    Batch Size: {config['batch_size']}, LR: {config['lr']}, "
                  f"Weight Decay: {config['weight_decay']}", flush=True)
            print(f"    Optimizer: {config['optimizer']}, Grad Clip: {config['grad_clip']}, "
                  f"LR Patience: {config['lr_patience']}", flush=True)

            # Train model
            result = train_model(
                config, train_df, val_df, feature_cols, device,
                max_epochs=max_epochs, patience=args.patience
            )

            # Combine config and results
            full_result = {**config, **result}
            full_result['config_id'] = i + 1
            full_result['training_time_sec'] = time.time() - config_start

            results.append(full_result)

            # Save results after EVERY config (not just every 10)
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('best_val_r2', ascending=False)
            results_df.to_csv(output_file, index=False)

            # Print summary
            print(f"\n{'='*70}", flush=True)
            print(f"Config {i+1}/{n_configs} Complete:", flush=True)
            print(f"  Best Val R²: {result['best_val_r2']:.6f} (epoch {result['best_epoch']})", flush=True)
            print(f"  Overfitting Gap: {result['overfitting_gap']:.6f}", flush=True)
            print(f"  Parameters: {result['n_parameters']:,}", flush=True)
            print(f"  Training Time: {full_result['training_time_sec']:.1f}s", flush=True)
            print(f"  Architecture: lookback={config['lookback']}, hidden={config['hidden_size']}, "
                  f"layers={config['num_layers']}, dropout={config['dropout']}", flush=True)
            print(f"  Output Head: fc_layers={config['fc_num_layers']}, "
                  f"fc_activation={config['fc_activation']}, fc_dropout={config['fc_dropout']}", flush=True)
            print(f"  Training: lr={config['lr']}, weight_decay={config['weight_decay']}, "
                  f"optimizer={config['optimizer']}", flush=True)
            print(f"  Results saved to: {output_file}", flush=True)

        except Exception as e:
            print(f"\nERROR in config {i+1}: {e}", flush=True)
            continue

    # Save final results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_val_r2', ascending=False)
    results_df.to_csv(output_file, index=False)

    elapsed = time.time() - start_time

    print(f"\n{'='*70}", flush=True)
    print("SEARCH COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Total time: {elapsed/60:.1f} minutes", flush=True)
    print(f"Successful configs: {len(results)}/{n_configs}", flush=True)
    print(f"Results saved: {output_file}", flush=True)
    print(flush=True)

    # Print top 5 configurations
    print("Top 5 configurations:", flush=True)
    print(results_df.head(5)[['config_id', 'best_val_r2', 'overfitting_gap',
                               'lookback', 'hidden_size', 'num_layers',
                               'dropout', 'n_parameters']].to_string(index=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
