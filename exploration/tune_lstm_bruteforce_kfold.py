#!/usr/bin/env python3
"""
K-Fold Brute Force Hyperparameter Search with Auto-Submission

Two-stage training:
1. K-fold CV on sampled data to find best hyperparameters
2. Full training on 100% data if k-fold mean R² beats global best

Features:
- K-fold cross-validation for robust hyperparameter selection
- Configurable sampling (--kfold_sample) for faster search
- Final training on full dataset for best configs
- Auto-generates submission when mean_val_r2 > best_global_r2
- Tracks best models across entire search

Usage:
  Test mode: python3 tune_lstm_bruteforce_kfold.py --test --kfold_sample 0.5
  Full mode (25% k-fold): python3 tune_lstm_bruteforce_kfold.py --kfold_sample 0.25
  Custom: python3 tune_lstm_bruteforce_kfold.py --n_configs 50 --kfold_sample 0.3 --n_folds 5
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
from sklearn.model_selection import KFold
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


def train_model_kfold(config, sampled_df, feature_cols, device, n_folds=5, max_epochs=20, patience=5, logger=None):
    """
    Train model using k-fold cross-validation on sampled data.

    Args:
        config: Hyperparameter configuration
        sampled_df: Sampled dataframe for k-fold
        feature_cols: Feature column names
        device: Training device
        n_folds: Number of folds (default: 5)
        max_epochs: Max epochs per fold
        patience: Early stopping patience
        logger: Logger instance

    Returns:
        dict with mean_val_r2, std_val_r2, min_val_r2, max_val_r2, fold_results
    """
    # Get unique sequences
    all_seqs = sorted(sampled_df['seq_ix'].unique())

    if logger:
        logger.info(f"  K-Fold CV: {n_folds} folds on {len(all_seqs)} sequences")

    # K-fold split
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    fold_times = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_seqs)):
        fold_start = time.time()

        # Get sequences for this fold
        train_seqs = [all_seqs[i] for i in train_idx]
        val_seqs = [all_seqs[i] for i in val_idx]

        # Create train/val dataframes
        train_fold_df = sampled_df[sampled_df['seq_ix'].isin(train_seqs)]
        val_fold_df = sampled_df[sampled_df['seq_ix'].isin(val_seqs)]

        # Train on this fold
        result = train_model(
            config, train_fold_df, val_fold_df, feature_cols, device,
            max_epochs=max_epochs, patience=patience, logger=logger
        )

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)

        fold_results.append({
            'fold': fold_idx + 1,
            'val_r2': result['best_val_r2'],
            'train_r2': result['best_train_r2'],
            'overfitting_gap': result['overfitting_gap'],
            'best_epoch': result['best_epoch'],
            'n_parameters': result['n_parameters'],
            'time_sec': fold_time
        })

        if logger:
            logger.info(f"    Fold {fold_idx+1}/{n_folds}: Val R²={result['best_val_r2']:.4f} "
                       f"Train R²={result['best_train_r2']:.4f} ({fold_time:.1f}s)")

    # Calculate statistics
    val_r2s = [f['val_r2'] for f in fold_results]
    mean_val_r2 = np.mean(val_r2s)
    std_val_r2 = np.std(val_r2s)
    min_val_r2 = np.min(val_r2s)
    max_val_r2 = np.max(val_r2s)

    if logger:
        logger.info(f"  K-Fold Results: mean={mean_val_r2:.6f} ± {std_val_r2:.6f} "
                   f"(min={min_val_r2:.6f}, max={max_val_r2:.6f})")

    # Get n_parameters from first fold result (all folds have same model architecture)
    n_parameters = fold_results[0].get('n_parameters', result.get('n_parameters', 0))

    return {
        'mean_val_r2': mean_val_r2,
        'std_val_r2': std_val_r2,
        'min_val_r2': min_val_r2,
        'max_val_r2': max_val_r2,
        'n_parameters': n_parameters,
        'fold_results': fold_results,
        'total_kfold_time': sum(fold_times)
    }


def train_final_model(config, full_train_df, full_val_df, feature_cols, device, max_epochs=30, patience=7, logger=None):
    """
    Train final model on 100% train/val split.

    Used when k-fold mean R² beats global best.

    Args:
        config: Hyperparameter configuration
        full_train_df: Full training set (80% of all sequences)
        full_val_df: Full validation set (20% of all sequences)
        feature_cols: Feature column names
        device: Training device
        max_epochs: Max epochs for final training
        patience: Early stopping patience
        logger: Logger instance

    Returns:
        dict with best_val_r2, best_model_state, and metrics
    """
    if logger:
        logger.info(f"\n  Training FINAL model on 100% data...")
        logger.info(f"    Train: {full_train_df['seq_ix'].nunique()} sequences, "
                   f"{len(full_train_df):,} rows")
        logger.info(f"    Val:   {full_val_df['seq_ix'].nunique()} sequences, "
                   f"{len(full_val_df):,} rows")

    result = train_model(
        config, full_train_df, full_val_df, feature_cols, device,
        max_epochs=max_epochs, patience=patience, logger=logger
    )

    if logger:
        logger.info(f"  Final Model: Val R²={result['best_val_r2']:.6f} "
                   f"(epoch {result['best_epoch']}/{result['total_epochs']})")

    return result


def create_submission(config, result, feature_cols, config_id, timestamp, logger):
    """
    Create a competition submission folder with solution.py and model weights.

    Following the structure of wunderfund/submissions/lstm_full_r2/
    """
    # Create submission directory
    r2_str = f"{result['best_val_r2']:.4f}".replace('.', '')
    submission_name = f"bruteforce_r2_{r2_str}_{timestamp}_cfg{config_id}"
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
{model_type} Solution - Brute Force Search
Config ID: {config_id}
Timestamp: {timestamp}

Best validation R²: {result['best_val_r2']:.6f}
Training R²: {result['best_train_r2']:.6f}
Overfitting gap: {result['overfitting_gap']:.6f}

Architecture:
- {model_type} ({direction})
- {config['num_layers']} layers, {config['hidden_size']} hidden units
- Lookback: {lookback} timesteps
- Dropout: {config['dropout']}
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


class FlexibleRNN(nn.Module):
    """Flexible RNN-based sequence predictor."""

    def __init__(
        self,
        input_size,
        hidden_size={config['hidden_size']},
        num_layers={config['num_layers']},
        dropout={config['dropout']},
        bidirectional={config['bidirectional']},
        use_gru={config['use_gru']},
        fc_num_layers={config['fc_num_layers']},
        fc_hidden_dims={config['fc_hidden_dims']},
        fc_activation='{config['fc_activation']}',
        fc_dropout={config['fc_dropout']},
        use_batch_norm={config['use_batch_norm']}
    ):
        super(FlexibleRNN, self).__init__()

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
        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        return self.output_head(last_output)


class PredictionModel:
    """
    Wrapper for {model_type} model matching competition API.

    Maintains lookback window and predicts next state.
    Falls back to simple average when insufficient history.
    """

    def __init__(self, lookback={lookback}, n_features={n_features}):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')  # Use CPU for submission

        # Initialize model
        self.model = FlexibleRNN(input_size=n_features).to(self.device)

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
    parser = argparse.ArgumentParser(description='K-fold brute force hyperparameter search with auto-submission')
    parser.add_argument('--test', action='store_true', help='Test mode (tiny data, 5 configs)')
    parser.add_argument('--n_configs', type=int, default=100, help='Number of configurations to test')
    parser.add_argument('--max_epochs', type=int, default=20, help='Max epochs for k-fold training')
    parser.add_argument('--final_max_epochs', type=int, default=30, help='Max epochs for final training')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience for k-fold')
    parser.add_argument('--final_patience', type=int, default=7, help='Early stopping patience for final training')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'mps', 'cuda', 'auto'],
                       help='Device to use for training (default: auto-detect)')
    parser.add_argument('--best_r2', type=float, default=0.34,
                       help='Starting best R² to beat (default: 0.34)')
    parser.add_argument('--kfold_sample', type=float, default=0.25,
                       help='Fraction of data to use for k-fold CV (default: 0.25)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--num_tune_before_train', type=int, default=20,
                       help='Number of configs to tune before training best from batch (default: 20)')
    parser.add_argument('--train_sample', type=float, default=1.0,
                       help='Fraction of full train data to use for final training (default: 1.0 = 100%%)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Setup logging
    logger, log_file = setup_logging(timestamp)

    logger.info("="*70)
    logger.info("K-FOLD BRUTE FORCE HYPERPARAMETER SEARCH WITH AUTO-SUBMISSION")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {'TEST' if args.test else 'FULL'}")
    logger.info(f"Best R² to beat: {args.best_r2:.6f}")
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
        full_df = pd.read_parquet('data/tiny_train.parquet')
        n_configs = 5
        max_epochs_kfold = 3
        max_epochs_final = 5
        n_folds = 2  # Override: tiny data has very few sequences
        kfold_sample = 0.8  # Use 80% for k-fold in test mode (more sequences)
        num_tune_before_train = 3  # Small batch for testing
        output_file = 'models/test_bruteforce_kfold_results.csv'
    else:
        full_df = pd.read_parquet('../competition_package/datasets/train.parquet')
        n_configs = args.n_configs
        max_epochs_kfold = args.max_epochs
        max_epochs_final = args.final_max_epochs
        n_folds = args.n_folds
        kfold_sample = args.kfold_sample
        num_tune_before_train = args.num_tune_before_train
        output_file = f'models/bruteforce_kfold_results_{timestamp}.csv'

    # Get all sequences
    all_seqs = sorted(full_df['seq_ix'].unique())

    # Split into full train/val (80/20) for final training
    n_train = int(0.8 * len(all_seqs))
    full_train_seqs = all_seqs[:n_train]
    full_val_seqs = all_seqs[n_train:]

    # Apply train_sample to the training sequences
    train_sample = args.train_sample if not args.test else 1.0  # Always use 100% in test mode
    if train_sample < 1.0:
        n_sampled_train = int(len(full_train_seqs) * train_sample)
        sampled_train_seqs = full_train_seqs[:n_sampled_train]  # Take first N to maintain temporal order
        final_train_df = full_df[full_df['seq_ix'].isin(sampled_train_seqs)]
    else:
        final_train_df = full_df[full_df['seq_ix'].isin(full_train_seqs)]

    final_val_df = full_df[full_df['seq_ix'].isin(full_val_seqs)]

    # Sample data for k-fold CV
    n_sample = int(len(all_seqs) * kfold_sample)
    sampled_seqs = np.random.choice(all_seqs, n_sample, replace=False)
    sampled_df = full_df[full_df['seq_ix'].isin(sampled_seqs)]

    feature_cols = [col for col in full_df.columns
                   if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]

    logger.info(f"Total sequences: {len(all_seqs)}")
    logger.info(f"K-Fold sample: {n_sample} sequences ({kfold_sample*100:.0f}%)")
    logger.info(f"Full train sequences (80%): {len(full_train_seqs)}")
    logger.info(f"Final train: {len(final_train_df['seq_ix'].unique())} sequences ({train_sample*100:.0f}%), {len(final_train_df):,} rows")
    logger.info(f"Final val: {len(full_val_seqs)} sequences, {len(final_val_df):,} rows")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Configurations to test: {n_configs}")
    logger.info(f"K-Fold: {n_folds} folds, max {max_epochs_kfold} epochs")
    logger.info(f"Final: max {max_epochs_final} epochs")
    logger.info("")

    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('../submissions').mkdir(exist_ok=True)

    # Initialize global best tracking
    best_global_r2 = args.best_r2
    submissions_log = []

    # Batch tracking for delayed training
    batch_candidates = []  # Store configs that beat global best in current batch
    num_tune_before_train = args.num_tune_before_train

    # Run search
    results = []
    start_time = time.time()

    logger.info("Starting brute force search...")
    logger.info(f"Batch size: Evaluate {num_tune_before_train} configs, then train best from batch")
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

            # STAGE 1: K-Fold Cross-Validation on Sampled Data
            logger.info(f"\n  Stage 1: Running {n_folds}-Fold CV on {kfold_sample*100:.0f}% sampled data...")
            kfold_result = train_model_kfold(
                config, sampled_df, feature_cols, device,
                n_folds=n_folds, max_epochs=max_epochs_kfold,
                patience=args.patience, logger=logger
            )

            # Combine config and k-fold results
            full_result = {**config, **kfold_result}
            full_result['config_id'] = i + 1

            # Print k-fold summary
            logger.info(f"\n{'='*70}")
            logger.info(f"Config {i+1}/{n_configs} - K-Fold Complete:")
            logger.info(f"  Mean Val R²: {kfold_result['mean_val_r2']:.6f} ± {kfold_result['std_val_r2']:.6f}")
            logger.info(f"  Range: [{kfold_result['min_val_r2']:.6f}, {kfold_result['max_val_r2']:.6f}]")
            logger.info(f"  Best Global Mean R²: {best_global_r2:.6f}")
            logger.info(f"  Parameters: {kfold_result['n_parameters']:,}")

            # STAGE 2: Check if this beats the global best
            if kfold_result['mean_val_r2'] > best_global_r2:
                logger.info(f"\n🎯 CANDIDATE! Mean R² ({kfold_result['mean_val_r2']:.6f}) > Global Best ({best_global_r2:.6f})")
                logger.info(f"  Adding to batch candidates (will train best after {num_tune_before_train} configs)")

                # Store candidate for batch processing
                batch_candidates.append({
                    'config_id': i + 1,
                    'config': config,
                    'kfold_result': kfold_result,
                    'config_start': config_start
                })

                full_result['final_val_r2'] = None
                full_result['final_train_r2'] = None
                full_result['final_overfitting_gap'] = None
                full_result['final_best_epoch'] = None
                full_result['final_total_epochs'] = None
                full_result['training_time_sec'] = time.time() - config_start
            else:
                logger.info(f"  K-Fold mean R² not better than global best ({best_global_r2:.6f})")
                logger.info(f"  Skipping.")
                full_result['final_val_r2'] = None
                full_result['final_train_r2'] = None
                full_result['final_overfitting_gap'] = None
                full_result['final_best_epoch'] = None
                full_result['final_total_epochs'] = None
                full_result['training_time_sec'] = time.time() - config_start

            # Don't save model state in CSV
            full_result_csv = {k: v for k, v in full_result.items() if k != 'best_model_state'}
            results.append(full_result_csv)

            # Save results dynamically after EVERY config
            results_df = pd.DataFrame(results)
            if len(results) > 0 and 'mean_val_r2' in results_df.columns:
                results_df_sorted = results_df.sort_values('mean_val_r2', ascending=False)
                results_df_sorted.to_csv(output_file, index=False)
            else:
                results_df.to_csv(output_file, index=False)
            logger.info(f"  Results saved to {output_file}")

            # Check if batch is complete
            if (i + 1) % num_tune_before_train == 0 or (i + 1) == n_configs:
                if len(batch_candidates) > 0:
                    logger.info(f"\n{'='*70}")
                    logger.info(f"BATCH COMPLETE: {len(batch_candidates)} candidates found")
                    logger.info(f"{'='*70}")

                    # Find best candidate from batch
                    best_candidate = max(batch_candidates, key=lambda x: x['kfold_result']['mean_val_r2'])
                    best_mean_r2 = best_candidate['kfold_result']['mean_val_r2']

                    logger.info(f"Best from batch: Config {best_candidate['config_id']} (Mean R²={best_mean_r2:.6f})")
                    logger.info(f"\n  Stage 2: Training best candidate on {train_sample*100:.0f}% train data...")

                    # Train final model on sampled train dataset
                    final_result = train_final_model(
                        best_candidate['config'], final_train_df, final_val_df, feature_cols, device,
                        max_epochs=max_epochs_final, patience=args.final_patience, logger=logger
                    )

                    # Print final summary
                    logger.info(f"\n{'='*70}")
                    logger.info(f"Config {best_candidate['config_id']}/{n_configs} - Final Training Complete:")
                    logger.info(f"  K-Fold Mean R²: {best_mean_r2:.6f}")
                    logger.info(f"  Final Val R²: {final_result['best_val_r2']:.6f} (epoch {final_result['best_epoch']})")
                    logger.info(f"  Final Train R²: {final_result['best_train_r2']:.6f}")
                    logger.info(f"  Overfitting Gap: {final_result['overfitting_gap']:.6f}")

                    # Update global best
                    improvement = best_mean_r2 - best_global_r2
                    best_global_r2 = best_mean_r2

                    logger.info(f"\n🎉🎉🎉 NEW GLOBAL BEST! R² improved by {improvement:.6f}")

                    # Create submission with final model
                    submission_dir = create_submission(
                        best_candidate['config'], final_result, feature_cols,
                        best_candidate['config_id'], timestamp, logger
                    )

                    # Log submission
                    submissions_log.append({
                        'config_id': best_candidate['config_id'],
                        'mean_val_r2_kfold': best_mean_r2,
                        'std_val_r2_kfold': best_candidate['kfold_result']['std_val_r2'],
                        'final_val_r2': final_result['best_val_r2'],
                        'improvement': improvement,
                        'submission_dir': submission_dir,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

                    # Save submissions log
                    submissions_df = pd.DataFrame(submissions_log)
                    submissions_df.to_csv('models/bruteforce_kfold_submissions_log.csv', index=False)

                    # Update the results entry for this config with final training results
                    for result_entry in results:
                        if result_entry['config_id'] == best_candidate['config_id']:
                            result_entry['final_val_r2'] = final_result['best_val_r2']
                            result_entry['final_train_r2'] = final_result['best_train_r2']
                            result_entry['final_overfitting_gap'] = final_result['overfitting_gap']
                            result_entry['final_best_epoch'] = final_result['best_epoch']
                            result_entry['final_total_epochs'] = final_result['total_epochs']
                            break

                    # Save updated results
                    results_df = pd.DataFrame(results)
                    if 'mean_val_r2' in results_df.columns:
                        results_df_sorted = results_df.sort_values('mean_val_r2', ascending=False)
                        results_df_sorted.to_csv(output_file, index=False)
                    logger.info(f"  Updated results saved to {output_file}")

                    # Clear batch for next round
                    batch_candidates = []
                else:
                    logger.info(f"\n{'='*70}")
                    logger.info(f"BATCH COMPLETE: No candidates beat global best ({best_global_r2:.6f})")
                    logger.info(f"{'='*70}")

        except Exception as e:
            logger.info(f"\nERROR in config {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save final results
    results_df = pd.DataFrame(results)
    if len(results) > 0 and 'mean_val_r2' in results_df.columns:
        results_df = results_df.sort_values('mean_val_r2', ascending=False)
    results_df.to_csv(output_file, index=False)

    elapsed = time.time() - start_time

    logger.info(f"\n{'='*70}")
    logger.info("SEARCH COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Successful configs: {len(results)}/{n_configs}")
    logger.info(f"Final best mean K-Fold R²: {best_global_r2:.6f}")
    logger.info(f"Submissions created: {len(submissions_log)}")
    logger.info(f"Results saved: {output_file}")
    logger.info("")

    # Print top 5 configurations
    logger.info("Top 5 configurations:")
    top_cols = ['config_id', 'mean_val_r2', 'std_val_r2', 'final_val_r2',
                'lookback', 'hidden_size', 'num_layers', 'dropout', 'bidirectional', 'use_gru',
                'fc_num_layers', 'fc_hidden_dims', 'fc_activation', 'fc_dropout', 'use_batch_norm',
                'batch_size', 'lr', 'weight_decay', 'grad_clip', 'optimizer', 'lr_patience',
                'n_parameters']
    available_cols = [col for col in top_cols if col in results_df.columns]
    logger.info("\n" + results_df.head(5)[available_cols].to_string(index=False))

    if len(submissions_log) > 0:
        logger.info(f"\nSubmissions log saved: models/bruteforce_kfold_submissions_log.csv")
        logger.info("\nSubmissions created:")
        for sub in submissions_log:
            logger.info(f"  {sub['submission_dir']} (K-Fold Mean R²={sub['mean_val_r2_kfold']:.6f}, Final R²={sub['final_val_r2']:.6f})")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
