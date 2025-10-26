#!/usr/bin/env python3
"""
Attention Architecture Hyperparameter Search with Auto-Submission

Focused search for GRU+Attention models using best GRU config as base.
Only varies attention-specific parameters (heads, dropout).

Features:
- Uses best GRU config (Val R² = 0.3566) as foundation
- Varies only: attention_heads [2, 4, 8] × attention_dropout [0.0, 0.1, 0.2, 0.3]
- Auto-generates submission when val R² > 0.3566
- Tracks best attention models

Usage:
  Test mode (tiny data, 5 configs): python3 tune_attention_bruteforce.py --test
  Full mode (100% data, 16 configs): python3 tune_attention_bruteforce.py
  Custom: python3 tune_attention_bruteforce.py --n_configs 12 --max_epochs 40
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
import shutil
import logging

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Add current directory to path
sys.path.append('.')
sys.path.append('../competition_package')

from models import create_model_from_config


def setup_logging(timestamp):
    """
    Setup dual logging to both console and file.

    Args:
        timestamp: Timestamp string for log filename

    Returns:
        Configured logger instance
    """
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger('bruteforce')
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # File handler
    log_file = f'logs/bruteforce_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


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
    """
    Sample hyperparameters for attention search.

    FROZEN: Uses best GRU config (Val R² = 0.3566) from metadata.json
    TUNING: Only attention_heads and attention_dropout

    Returns:
        Dictionary of hyperparameters
    """
    search_space = {
            # Architecture
            'lookback': [20, 30, 50],
            'hidden_size': [64, 128, 192],
            'num_layers': [1, 2, 4],
            'dropout': [0.1, 0.2, 0.3],
            'bidirectional': [False, True],
            'use_gru': [True, False],

            # Output head
            'fc_num_layers': [2, 3],
            'fc_hidden_dims': [64, 128, 256],
            'fc_activation': ['relu', 'tanh', 'gelu'],
            'fc_dropout': [0.0, 0.2, 0.3],
            'use_batch_norm': [True, False],

            # Training
            'batch_size': [128, 256],
            'lr': [1e-4, 5e-4, 1e-3],
            'weight_decay': [0.0, 1e-4, 1e-3],
            'grad_clip': [None, 0.5, 1.0],
            'optimizer': ['adam', 'adamw'],
            'lr_patience': [3],
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


def train_model(config, train_df, val_df, feature_cols, device, max_epochs=20, patience=5, logger=None):
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
    best_model_state = None
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
        msg = f"    Epoch {epoch+1:2d}/{max_epochs}: Val R²={val_r2:.4f} Train R²={train_r2:.4f} Gap={gap:.4f} {marker}"
        if logger:
            logger.info(msg)
        else:
            print(msg, flush=True)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"    Early stopped at epoch {epoch+1}")
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
        'best_model_state': best_model_state,
        'history': history
    }


def create_submission(config, result, feature_cols, config_id, timestamp, logger):
    """
    Create a competition submission folder with solution.py and model weights.

    Following the structure of wunderfund/submissions/lstm_full_r2/
    """
    # Create submission directory
    r2_str = f"{result['best_val_r2']:.4f}".replace('.', '')
    submission_name = f"attention_r2_{r2_str}_{timestamp}_cfg{config_id}"
    submission_dir = Path(f"../submissions/{submission_name}")
    submission_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"🏆 CREATING SUBMISSION - NEW BEST MODEL!")
    logger.info(f"{'='*70}")
    logger.info(f"  Val R²: {result['best_val_r2']:.6f}")
    logger.info(f"  Directory: {submission_dir}")

    # Save model checkpoint
    model_filename = f"model_best.pt"
    model_path = submission_dir / model_filename

    # Create checkpoint
    checkpoint = {
        'model_state_dict': result['best_model_state'],
        'config': config,
        'val_r2': result['best_val_r2'],
        'train_r2': result['best_train_r2'],
        'best_epoch': result['best_epoch'],
        'n_parameters': result['n_parameters'],
        'feature_cols': feature_cols,
        'timestamp': timestamp
    }
    torch.save(checkpoint, model_path)
    logger.info(f"  Saved model: {model_filename}")

    # Generate solution.py
    n_features = len(feature_cols)
    lookback = config['lookback']

    # Determine model type
    model_type = "GRU" if config['use_gru'] else "LSTM"
    direction = "Bidirectional" if config['bidirectional'] else "Unidirectional"

    solution_code = f'''"""
{model_type}+Attention Solution - Attention Hyperparameter Search
Config ID: {config_id}
Timestamp: {timestamp}

Best validation R²: {result['best_val_r2']:.6f}
Training R²: {result['best_train_r2']:.6f}
Overfitting gap: {result['overfitting_gap']:.6f}

Architecture:
- Base: {model_type} ({direction})
  - {config['num_layers']} layers, {config['hidden_size']} hidden units
  - Lookback: {lookback} timesteps
  - Dropout: {config['dropout']}
- Attention: Multi-Head
  - Heads: {config['attention_heads']}
  - Dropout: {config['attention_dropout']}
- Output Head:
  - FC layers: {config['fc_num_layers']}, activation: {config['fc_activation']}
- Parameters: {result['n_parameters']:,}

Training:
- Optimizer: {config['optimizer'].upper()}
- Learning rate: {config['lr']}
- Weight decay: {config['weight_decay']}
- Batch size: {config['batch_size']}
- Best epoch: {result['best_epoch']}/{result['total_epochs']}
"""

import numpy as np
import torch
import torch.nn as nn
from utils import DataPoint


class RNNWithAttention(nn.Module):
    """RNN + Multi-Head Attention hybrid architecture."""

    def __init__(
        self,
        input_size,
        hidden_size={config['hidden_size']},
        num_layers={config['num_layers']},
        dropout={config['dropout']},
        bidirectional={config['bidirectional']},
        use_gru={config['use_gru']},
        attention_heads={config['attention_heads']},
        attention_dropout={config['attention_dropout']},
        fc_num_layers={config['fc_num_layers']},
        fc_hidden_dims={config['fc_hidden_dims']},
        fc_activation='{config['fc_activation']}',
        fc_dropout={config['fc_dropout']},
        use_batch_norm={config['use_batch_norm']}
    ):
        super(RNNWithAttention, self).__init__()

        # RNN layer
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # RNN output size
        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=rnn_output_size,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(rnn_output_size)
        self.attention_dropout = nn.Dropout(attention_dropout)

        # Build output head
        self.output_head = self._build_output_head(
            rnn_output_size, input_size, fc_num_layers,
            fc_hidden_dims, fc_activation, fc_dropout, use_batch_norm
        )

    def _build_output_head(self, input_dim, output_dim, num_layers,
                           hidden_dims, activation, dropout, use_batch_norm):
        """Build fully connected output head."""
        layers = []

        activation_map = {{
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2)
        }}
        act_fn = activation_map.get(activation, nn.ReLU())

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            current_dim = input_dim
            for i in range(num_layers - 1):
                layers.append(nn.Linear(current_dim, hidden_dims))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dims))
                layers.append(act_fn)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                current_dim = hidden_dims
            layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        # Apply attention
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        rnn_out = self.attention_norm(rnn_out + self.attention_dropout(attn_out))
        # Take last timestep
        last_output = rnn_out[:, -1, :]
        return self.output_head(last_output)


class PredictionModel:
    """
    Wrapper for {model_type}+Attention model matching competition API.

    Maintains lookback window and predicts next state.
    Falls back to simple average when insufficient history.
    """

    def __init__(self, lookback={lookback}, n_features={n_features}):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')  # Use CPU for submission

        # Initialize model (RNN with Attention)
        self.model = RNNWithAttention(input_size=n_features).to(self.device)

        # Load trained weights
        checkpoint = torch.load('{model_filename}', map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Sequence state
        self.current_seq_ix = None
        self.sequence_history = []

    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Generate prediction for next timestep.

        Args:
            data_point: Current observation

        Returns:
            None if prediction not needed, otherwise np.ndarray of predicted next state
        """
        # Reset on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Add current state to history
        self.sequence_history.append(data_point.state.copy())

        if not data_point.need_prediction:
            return None

        # Fallback to average if not enough history
        if len(self.sequence_history) < self.lookback:
            return np.mean(self.sequence_history, axis=0)

        # Use model for prediction
        sequence = np.array(
            self.sequence_history[-self.lookback:],
            dtype=np.float32
        )
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(sequence_tensor)
            prediction = prediction.cpu().numpy()[0]

        return prediction
'''

    # Write solution.py
    solution_path = submission_dir / 'solution.py'
    with open(solution_path, 'w') as f:
        f.write(solution_code)
    logger.info(f"  Generated: solution.py")

    # Copy utils.py from competition package
    utils_src = Path('../competition_package/utils.py')
    utils_dst = submission_dir / 'utils.py'
    shutil.copy(utils_src, utils_dst)
    logger.info(f"  Copied: utils.py")

    # Save metadata
    metadata = {
        'config_id': config_id,
        'timestamp': timestamp,
        'val_r2': result['best_val_r2'],
        'train_r2': result['best_train_r2'],
        'overfitting_gap': result['overfitting_gap'],
        'best_epoch': result['best_epoch'],
        'total_epochs': result['total_epochs'],
        'n_parameters': result['n_parameters'],
        'config': config
    }

    metadata_path = submission_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved: metadata.json")

    logger.info(f"{'='*70}\n")

    return str(submission_dir)


def main():
    parser = argparse.ArgumentParser(description='GRU+Attention hyperparameter search with auto-submission')
    parser.add_argument('--test', action='store_true', help='Test mode (tiny data, 5 configs)')
    parser.add_argument('--n_configs', type=int, default=16, help='Number of configurations to test (default: 16 for 3x4 grid)')
    parser.add_argument('--max_epochs', type=int, default=40, help='Max epochs per config')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'mps', 'cuda', 'auto'],
                       help='Device to use for training (default: auto-detect)')
    parser.add_argument('--best_r2', type=float, default=0.3566,
                       help='Starting best R² to beat (default: 0.3566 from best GRU)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Setup logging
    logger, log_file = setup_logging(timestamp)

    logger.info("="*70)
    logger.info("GRU+ATTENTION HYPERPARAMETER SEARCH WITH AUTO-SUBMISSION")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {'TEST (tiny)' if args.test else 'FULL (100%)'}")
    logger.info(f"Base: Best GRU config (Val R² = {args.best_r2:.6f})")
    logger.info(f"Tuning: attention_heads [2, 4, 8] × attention_dropout [0.0, 0.1, 0.2, 0.3]")
    logger.info(f"Target: Beat baseline R² > {args.best_r2:.6f}")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    # Detect device
    device_str, has_mps, has_cuda = detect_device(args.device)
    device = torch.device(device_str)

    logger.info("Device Detection:")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  Requested: {args.device}")
    logger.info(f"  MPS available: {'✓' if has_mps else '✗'}")
    logger.info(f"  CUDA available: {'✓' if has_cuda else '✗'}")
    logger.info(f"  Using: {device_str.upper()}")
    logger.info("")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    logger.info("Loading data...")
    if args.test:
        train_df = pd.read_parquet('data/tiny_train.parquet')
        val_df = pd.read_parquet('data/tiny_val.parquet')
        n_configs = 5
        max_epochs = 5
        output_file = 'models/test_attention_search_results.csv'
    else:
        full_df = pd.read_parquet('../competition_package/datasets/train.parquet')
        all_seqs = full_df['seq_ix'].unique()

        # Use 100% of data - split 80/20 train/val
        n_train = int(0.8 * len(all_seqs))
        train_seqs = all_seqs[:n_train]
        val_seqs = all_seqs[n_train:]

        train_df = full_df[full_df['seq_ix'].isin(train_seqs)]
        val_df = full_df[full_df['seq_ix'].isin(val_seqs)]

        n_configs = args.n_configs
        max_epochs = args.max_epochs
        output_file = f'models/attention_search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    feature_cols = [col for col in train_df.columns
                   if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]

    logger.info(f"Train: {len(train_df):,} rows, {train_df['seq_ix'].nunique()} sequences")
    logger.info(f"Val:   {len(val_df):,} rows, {val_df['seq_ix'].nunique()} sequences")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Configurations to test: {n_configs}")
    logger.info(f"Max epochs per config: {max_epochs}")
    logger.info("")

    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('../submissions').mkdir(exist_ok=True)

    # Initialize global best tracking
    best_global_r2 = args.best_r2
    submissions_log = []

    # Run search
    results = []
    start_time = time.time()

    logger.info("Starting brute force search...")
    logger.info("")

    for i in tqdm(range(n_configs), desc="Searching"):
        config_start = time.time()

        # Sample hyperparameters
        config = sample_hyperparameters()

        try:
            # Print config details before training
            logger.info(f"\n{'='*70}")
            logger.info(f"Config {i+1}/{n_configs} - Starting Training:")
            logger.info(f"  Architecture:")
            logger.info(f"    Lookback: {config['lookback']}, Hidden: {config['hidden_size']}, "
                  f"Layers: {config['num_layers']}, Dropout: {config['dropout']}")
            logger.info(f"    Bidirectional: {config['bidirectional']}, GRU: {config['use_gru']}")
            logger.info(f"  Output Head:")
            logger.info(f"    FC Layers: {config['fc_num_layers']}, FC Hidden: {config['fc_hidden_dims']}, "
                  f"Activation: {config['fc_activation']}")
            logger.info(f"    FC Dropout: {config['fc_dropout']}, Batch Norm: {config['use_batch_norm']}")
            logger.info(f"  Training:")
            logger.info(f"    Batch Size: {config['batch_size']}, LR: {config['lr']}, "
                  f"Weight Decay: {config['weight_decay']}")
            logger.info(f"    Optimizer: {config['optimizer']}, Grad Clip: {config['grad_clip']}, "
                  f"LR Patience: {config['lr_patience']}")

            # Train model
            result = train_model(
                config, train_df, val_df, feature_cols, device,
                max_epochs=max_epochs, patience=args.patience, logger=logger
            )

            # Combine config and results
            full_result = {**config, **result}
            full_result['config_id'] = i + 1
            full_result['training_time_sec'] = time.time() - config_start
            # Don't save model state in CSV
            full_result_csv = {k: v for k, v in full_result.items() if k != 'best_model_state'}
            results.append(full_result_csv)

            # Print summary
            logger.info(f"\n{'='*70}")
            logger.info(f"Config {i+1}/{n_configs} Complete:")
            logger.info(f"  Best Val R²: {result['best_val_r2']:.6f} (epoch {result['best_epoch']})")
            logger.info(f"  Best Global R²: {best_global_r2:.6f}")
            logger.info(f"  Overfitting Gap: {result['overfitting_gap']:.6f}")
            logger.info(f"  Parameters: {result['n_parameters']:,}")
            logger.info(f"  Training Time: {full_result['training_time_sec']:.1f}s")

            # Check if this beats the global best
            if result['best_val_r2'] > best_global_r2:
                logger.info(f"\n🎉🎉🎉 NEW GLOBAL BEST! R² improved from {best_global_r2:.6f} to {result['best_val_r2']:.6f}")

                # Update global best
                improvement = result['best_val_r2'] - best_global_r2
                best_global_r2 = result['best_val_r2']

                # Create submission
                submission_dir = create_submission(
                    config, result, feature_cols, i + 1, timestamp, logger
                )

                # Log submission
                submissions_log.append({
                    'config_id': i + 1,
                    'val_r2': result['best_val_r2'],
                    'improvement': improvement,
                    'submission_dir': submission_dir,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

                # Save submissions log
                submissions_df = pd.DataFrame(submissions_log)
                submissions_df.to_csv('models/attention_submissions_log.csv', index=False)
            else:
                logger.info(f"  Not better than global best ({best_global_r2:.6f})")

            # Save results dynamically after EVERY config
            results_df = pd.DataFrame(results)
            results_df_sorted = results_df.sort_values('best_val_r2', ascending=False)
            results_df_sorted.to_csv(output_file, index=False)
            logger.info(f"  Results saved to {output_file}")

        except Exception as e:
            logger.info(f"\nERROR in config {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save final results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_val_r2', ascending=False)
    results_df.to_csv(output_file, index=False)

    elapsed = time.time() - start_time

    logger.info(f"\n{'='*70}")
    logger.info("SEARCH COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Successful configs: {len(results)}/{n_configs}")
    logger.info(f"Final best R²: {best_global_r2:.6f}")
    logger.info(f"Submissions created: {len(submissions_log)}")
    logger.info(f"Results saved: {output_file}")
    logger.info("")

    # Print top 5 configurations
    logger.info("Top 5 configurations:")
    logger.info("\n" + results_df.head(5)[['config_id', 'best_val_r2', 'overfitting_gap',
                               'lookback', 'hidden_size', 'num_layers',
                               'dropout', 'n_parameters']].to_string(index=False))

    if len(submissions_log) > 0:
        logger.info(f"\nSubmissions log saved: models/attention_submissions_log.csv")
        logger.info("\nSubmissions created:")
        for sub in submissions_log:
            logger.info(f"  {sub['submission_dir']} (R²={sub['val_r2']:.6f})")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
