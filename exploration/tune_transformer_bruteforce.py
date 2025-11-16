#!/usr/bin/env python3
"""
Brute Force Hyperparameter Search with Transformer Model

This script mirrors the behavior of `tune_lstm_bruteforce.py` but swaps the RNN (GRU/LSTM)
backbone for a lightweight Transformer encoder-based sequence predictor.

Usage is intentionally kept compatible with the original script so it can be used
in the same workflows (exhaustive config iterator, submission generation).
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
import itertools
import argparse
from datetime import datetime
from sklearn.metrics import r2_score
import random
from tqdm import tqdm
import shutil
import logging
import warnings
warnings.filterwarnings("ignore")

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Add current directory to path
sys.path.append('.')
sys.path.append('../competition_package')

def setup_logging(timestamp):
    Path('logs').mkdir(exist_ok=True)
    logger = logging.getLogger('bruteforce_transformer')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    log_file = f'logs/bruteforce_transformer_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file


class SequenceDataset(Dataset):
    """Creates sequences for training (same logic as the LSTM script)."""
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # last column remains zero if odd
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Lightweight Transformer encoder-based sequence predictor.
    Input: (batch, seq_len, n_features) -> Output: (batch, n_features)
    """
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2, dim_feedforward=256,
                 dropout=0.1, fc_num_layers=2, fc_hidden_dims=256, fc_activation='relu', use_batch_norm=True):
        super(TransformerPredictor, self).__init__()

        # Project input to d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        rnn_output_size = d_model

        # Build fully connected head
        self.output_head = self._build_output_head(rnn_output_size, input_size, fc_num_layers,
                                                   fc_hidden_dims, fc_activation, dropout, use_batch_norm)

    def _build_output_head(self, input_dim, output_dim, num_layers,
                           hidden_dims, activation, dropout, use_batch_norm):
        layers = []
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2)
        }
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
        # x: (batch, seq_len, features)
        x = self.input_proj(x)  # Project input to d_model dimensions
        x = self.pos_enc(x)     # Add positional encoding
        # Don't need permute since we use batch_first=True
        encoded = self.transformer(x)  # Get full sequence encoding
        last = encoded[:, -1, :]  # Get last timestep output for prediction
        out = self.output_head(last)  # Project to output size
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def detect_device(device_arg):
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    has_cuda = torch.cuda.is_available()
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
    n_features = len(feature_cols)

    train_dataset = SequenceDataset(train_df, config['lookback'], feature_cols)
    val_dataset = SequenceDataset(val_df, config['lookback'], feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Build transformer model from config
    model = TransformerPredictor(
        input_size=n_features,
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_transformer_layers', 2),
        dim_feedforward=config.get('dim_feedforward', 256),
        dropout=config.get('dropout', 0.1),
        fc_num_layers=config.get('fc_num_layers', 2),
        fc_hidden_dims=config.get('fc_hidden_dims', 256),
        fc_activation=config.get('fc_activation', 'relu'),
        use_batch_norm=config.get('use_batch_norm', True)
    ).to(device)

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config['lr_patience'])

    criterion = nn.MSELoss()

    best_val_r2 = -float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    history = []

    for epoch in range(max_epochs):
        train_loss, train_r2 = train_epoch(model, train_loader, criterion, optimizer, device, config['grad_clip'])
        val_loss, val_r2 = validate(model, val_loader, criterion, device)

        scheduler.step(val_r2)

        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_r2': train_r2, 'val_loss': val_loss, 'val_r2': val_r2, 'lr': optimizer.param_groups[0]['lr']})

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
            if logger:
                logger.info(f"    Early stopped at epoch {epoch+1}")
            break

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
    r2_str = f"{result['best_val_r2']:.4f}".replace('.', '')
    submission_name = f"bruteforce_transformer_r2_{r2_str}_{timestamp}_cfg{config_id}"
    submission_dir = Path(f"../submissions/{submission_name}")
    submission_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"🏆 CREATING TRANSFORMER SUBMISSION - NEW BEST MODEL!")
    logger.info(f"{'='*70}")
    logger.info(f"  Val R²: {result['best_val_r2']:.6f}")
    logger.info(f"  Directory: {submission_dir}")

    model_filename = f"model_best.pt"
    model_path = submission_dir / model_filename

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

    # Generate solution.py for submission (uses Transformer predictor)
    n_features = len(feature_cols)
    lookback = config['lookback']

    # Build submission script that exactly matches TransformerPredictor architecture
    solution_template = '''#!/usr/bin/env python3
"""
Transformer Solution - Brute Force Search
Config ID: __CONFIG_ID__
Timestamp: __TIMESTAMP__

Best validation R²: __VAL_R2__
Training R²: __TRAIN_R2__
Overfitting gap: __OVERFITTING_GAP__

Architecture:
- Transformer encoder (d_model=__D_MODEL__, layers=__NUM_LAYERS__, nhead=__NHEAD__)
- Lookback: __LOOKBACK__ timesteps
- Parameters: __N_PARAMS__

Training:
- Optimizer: __OPT__
- Learning rate: __LR__
- Weight decay: __WD__
- Batch size: __BATCH__
- Best epoch: __BEST_EPOCH__/__TOTAL_EPOCHS__
"""

import numpy as np
import torch
import torch.nn as nn
from utils import DataPoint


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FlexibleTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, num_layers=2, nhead=4, dim_feedforward=256, 
                 dropout=0.1, fc_hidden_dims=128, fc_activation='relu'):
        super(FlexibleTransformer, self).__init__()
        # Input projection and positional encoding (same as TransformerPredictor)
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder (using batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # For better performance
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Build output head matching TransformerPredictor architecture
        act_fn = nn.ReLU() if fc_activation == 'relu' else nn.GELU()
        self.output_head = nn.Sequential(
            nn.Linear(d_model, fc_hidden_dims),
            nn.BatchNorm1d(fc_hidden_dims),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dims, input_size)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        out = self.transformer(x)
        last = out[:, -1, :]
        return self.output_head(last)


class PredictionModel:
    def __init__(self, lookback=__LOOKBACK__, n_features=__N_FEATURES__):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')
        self.model = FlexibleTransformer(input_size=n_features, d_model=__D_MODEL__, num_layers=__NUM_LAYERS__, nhead=__NHEAD__)

        # Configure the model for inference
        self.device = torch.device('cpu')  # Force CPU for production deployments
        self.model.to(self.device)
        self.model.eval()

        # Load checkpoint (handle both state_dict and model_state_dict keys)
        try:
            checkpoint = torch.load('__MODEL_FILENAME__', map_location=self.device)
            state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
            self.model.load_state_dict(checkpoint[state_key], strict=True)
        except Exception as e:
            raise RuntimeError(f'Failed to load model checkpoint: {e}') from e

        # Initialize prediction state
        self.current_seq_ix = None  # Track sequence boundaries
        self.sequence_history = []   # Store lookback window

    def predict(self, data_point: DataPoint):
        # Handle sequence boundaries
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []  # Reset history on new sequence
        
        # Always store the current state
        self.sequence_history.append(data_point.state.copy())
        
        # Only predict when requested
        if not data_point.need_prediction:
            return None
            
        # Return mean of history for initial timesteps
        if len(self.sequence_history) < self.lookback:
            return np.mean(self.sequence_history, axis=0)
            
        # Prepare input sequence using lookback window
        sequence = np.array(self.sequence_history[-self.lookback:], dtype=np.float32)
        
        # Convert to tensor and add batch dimension
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            pred = self.model(seq_tensor)
            return pred.cpu().numpy()[0]  # Convert back to numpy array
'''

    # Fill placeholders
    solution_code = (solution_template
                     .replace('__CONFIG_ID__', str(config_id))
                     .replace('__TIMESTAMP__', timestamp)
                     .replace('__VAL_R2__', f"{result['best_val_r2']:.6f}")
                     .replace('__TRAIN_R2__', f"{result['best_train_r2']:.6f}")
                     .replace('__OVERFITTING_GAP__', f"{result['overfitting_gap']:.6f}")
                     .replace('__D_MODEL__', str(config.get('d_model', 128)))
                     .replace('__NUM_LAYERS__', str(config.get('num_transformer_layers', 2)))
                     .replace('__NHEAD__', str(config.get('nhead', 8)))
                     .replace('__LOOKBACK__', str(lookback))
                     .replace('__N_PARAMS__', f"{result['n_parameters']:,}")
                     .replace('__OPT__', config.get('optimizer', 'adam').upper())
                     .replace('__LR__', str(config.get('lr')))
                     .replace('__WD__', str(config.get('weight_decay')))
                     .replace('__BATCH__', str(config.get('batch_size')))
                     .replace('__BEST_EPOCH__', str(result['best_epoch']))
                     .replace('__TOTAL_EPOCHS__', str(result['total_epochs']))
                     .replace('__N_FEATURES__', str(n_features))
                     .replace('__MODEL_FILENAME__', model_filename)
                     )

    solution_path = submission_dir / 'solution.py'
    with open(solution_path, 'w') as f:
        f.write(solution_code)
    logger.info(f"  Generated: solution.py")

    utils_src = Path('../competition_package/utils.py')
    utils_dst = submission_dir / 'utils.py'
    shutil.copy(utils_src, utils_dst)
    logger.info(f"  Copied: utils.py")

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
    parser = argparse.ArgumentParser(description='Brute force hyperparameter search with Transformer backbone')
    parser.add_argument('--test', action='store_true', help='Test mode (tiny data, 3 configs)')
    parser.add_argument('--n_configs', type=int, default=100, help='Number of configurations to test (cap for exhaustive mode)')
    parser.add_argument('--max_epochs', type=int, default=40, help='Max epochs per config')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'mps', 'cuda', 'auto'])
    parser.add_argument('--best_r2', type=float, default=0.3560)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--max_configs', type=int, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger, log_file = setup_logging(timestamp)

    logger.info('='*70)
    logger.info('BRUTE FORCE TRANSFORMER SEARCH')
    logger.info('='*70)

    device_str, has_mps, has_cuda = detect_device(args.device)
    device = torch.device(device_str)

    logger.info(f"Using device: {device_str}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    logger.info('Loading data...')
    if args.test:
        train_df = pd.read_parquet('data/tiny_train.parquet')
        val_df = pd.read_parquet('data/tiny_val.parquet')
        effective_max_configs = 1000
        effective_max_epochs = 30
        output_file = 'models/test_transformer_bruteforce_search_results.csv'
    else:
        full_df = pd.read_parquet('../competition_package/datasets/train.parquet')
        all_seqs = full_df['seq_ix'].unique()
        n_train = int(0.8 * len(all_seqs))
        train_seqs = all_seqs[:n_train]
        val_seqs = all_seqs[n_train:]
        train_df = full_df[full_df['seq_ix'].isin(train_seqs)]
        val_df = full_df[full_df['seq_ix'].isin(val_seqs)]
        effective_max_configs = args.n_configs
        effective_max_epochs = args.max_epochs
        output_file = f'models/transformer_bruteforce_search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    feature_cols = [col for col in train_df.columns if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]

    logger.info(f"Train: {len(train_df):,} rows, {train_df['seq_ix'].nunique()} sequences")
    logger.info(f"Val:   {len(val_df):,} rows, {val_df['seq_ix'].nunique()} sequences")
    logger.info(f"Features: {len(feature_cols)}")

    Path('models').mkdir(exist_ok=True)
    Path('../submissions').mkdir(exist_ok=True)

    best_global_r2 = args.best_r2
    submissions_log = []

    # Simple search space (can be expanded)
    search_space = {
        # --- Sequence window ---
        'lookback': [50],

        # --- Transformer backbone ---
        'd_model': [128, 192, 256],             # model dimensionality (bigger → more capacity)
        'nhead': [4, 8],                        # 4 works well up to d_model=256
        'num_transformer_layers': [2, 3, 4],    # deeper = more expressive, but slower
        'dropout': [0.1, 0.2, 0.3],             # regularization for attention + FFN

        # --- Output head ---
        'fc_num_layers': [2, 3],
        'fc_hidden_dims': [128, 256, 512],
        'fc_activation': ['relu', 'gelu'],      # GELU sometimes helps transformer FFNs
        'use_batch_norm': [True, False],

        # --- Training hyperparams ---
        'batch_size': [128, 256, 512],
        'lr': [5e-5, 1e-4, 3e-4, 1e-3],         # transformers often prefer lower LR
        'weight_decay': [0.0, 1e-5, 1e-4],
        'grad_clip': [None, 0.5, 1.0],
        'optimizer': ['adam', 'adamw'],         # adamw often performs better with weight decay
        'lr_patience': [3, 5]
    }

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    logger.info(f"Total possible combinations (Cartesian product): {total_combinations:,}")

    if args.test:
        max_to_run = effective_max_configs
    else:
        if args.max_configs is not None:
            max_to_run = min(args.max_configs, total_combinations)
        else:
            max_to_run = min(effective_max_configs, total_combinations)
    logger.info(f"Configurations to run (cap): {max_to_run}")

    product_iter = itertools.product(*values)

    if args.shuffle:
        limited = list(itertools.islice(product_iter, max_to_run))
        random.seed(42)
        random.shuffle(limited)
        configs_iterable = (dict(zip(keys, tup)) for tup in limited)
    else:
        configs_iterable = (dict(zip(keys, tup)) for tup in itertools.islice(product_iter, max_to_run))

    results = []
    start_time = time.time()

    logger.info('Starting transformer brute force search...')

    for i, config in enumerate(tqdm(configs_iterable, total=max_to_run, desc='Searching'), start=1):
        config_start = time.time()
        config['batch_size'] = int(config['batch_size'])
        config['lr_patience'] = int(config.get('lr_patience', 3))

        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Config {i}/{max_to_run} - Starting Training:")
            logger.info(f"  Lookback: {config['lookback']}, d_model: {config['d_model']}, nhead: {config['nhead']}")

            result = train_model(config, train_df, val_df, feature_cols, device, max_epochs=effective_max_epochs, patience=args.patience, logger=logger)

            full_result = {**config, **result}
            full_result['config_id'] = i
            full_result['training_time_sec'] = time.time() - config_start
            full_result_csv = {k: v for k, v in full_result.items() if k != 'best_model_state'}
            results.append(full_result_csv)

            logger.info(f"\n{'='*70}")
            logger.info(f"Config {i}/{max_to_run} Complete:")
            logger.info(f"  Best Val R²: {result['best_val_r2']:.6f} (epoch {result['best_epoch']})")
            logger.info(f"  Best Global R²: {best_global_r2:.6f}")

            if result['best_val_r2'] > best_global_r2:
                logger.info(f"\n🎉 NEW GLOBAL BEST! R² improved from {best_global_r2:.6f} to {result['best_val_r2']:.6f}")
                improvement = result['best_val_r2'] - best_global_r2
                best_global_r2 = result['best_val_r2']
                submission_dir = create_submission(config, result, feature_cols, i, timestamp, logger)
                submissions_log.append({'config_id': i, 'val_r2': result['best_val_r2'], 'improvement': improvement, 'submission_dir': submission_dir, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                submissions_df = pd.DataFrame(submissions_log)
                submissions_df.to_csv('models/transformer_bruteforce_submissions_log.csv', index=False)
            else:
                logger.info(f"  Not better than global best ({best_global_r2:.6f})")

            results_df = pd.DataFrame(results)
            results_df_sorted = results_df.sort_values('best_val_r2', ascending=False)
            results_df_sorted.to_csv(output_file, index=False)
            logger.info(f"  Results saved to {output_file}")

        except Exception as e:
            logger.info(f"\nERROR in config {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('best_val_r2', ascending=False)
        results_df.to_csv(output_file, index=False)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info('SEARCH COMPLETE')
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Successful configs: {len(results)}/{max_to_run}")
    logger.info(f"Final best R²: {best_global_r2:.6f}")
    logger.info(f"Submissions created: {len(submissions_log)}")
    logger.info(f"Results saved: {output_file}")

    if not results_df.empty:
        logger.info('Top 5 configurations:')
        logger.info('\n' + results_df.head(5)[['config_id', 'best_val_r2', 'overfitting_gap', 'lookback', 'd_model', 'num_transformer_layers', 'nhead', 'n_parameters']].to_string(index=False))

    if len(submissions_log) > 0:
        logger.info(f"\nSubmissions log saved: models/transformer_bruteforce_submissions_log.csv")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
