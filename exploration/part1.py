#!/usr/bin/env python3
"""
Brute Force Hyperparameter Search with Auto-Submission
(using the new Encoder-Decoder LSTM architecture via create_model_from_config_new)

Features preserved from original:
- Exhaustive search / test mode
- Auto-submission when a new global best R² is found
- Logging to console + file
- CSV result saving after each config
- Early stopping, LR scheduler, grad clipping support
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

# Force unbuffered output (keep behavior from original)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Add current directory and competition package to path
sys.path.append('.')
sys.path.append('../competition_package')

# Use the new factory (the stored Encoder-Decoder implementation)
from models import create_model_from_config_new

# ---------------------------
# Logging
# ---------------------------
def setup_logging(timestamp):
    Path('logs').mkdir(exist_ok=True)
    logger = logging.getLogger('bruteforce')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    log_file = f'logs/bruteforce_{timestamp}.log'
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

# ---------------------------
# Dataset
# ---------------------------
class SequenceDataset(Dataset):
    """Creates sequences for RNN training."""
    def __init__(self, df, lookback, feature_cols):
        self.sequences = []
        self.targets = []

        # iterate over sequences
        for seq_ix in df['seq_ix'].unique():
            seq_data = df[df['seq_ix'] == seq_ix].sort_values('step_in_seq')
            features = seq_data[feature_cols].values

            for i in range(lookback, len(features)):
                self.sequences.append(features[i - lookback:i])
                self.targets.append(features[i])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

# ---------------------------
# Metrics / train / validate
# ---------------------------
def calculate_r2(predictions, targets):
    """Calculate mean R² across features."""
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()

    r2_scores = []
    for i in range(predictions.shape[1]):
        try:
            r2 = r2_score(targets[:, i], predictions[:, i])
        except Exception:
            r2 = -1.0
        r2_scores.append(r2)
    return np.mean(r2_scores)

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    model.train()
    total_loss = 0.0
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

    avg_loss = total_loss / max(1, len(loader))
    all_preds = torch.cat(all_preds, dim=0) if len(all_preds) > 0 else torch.zeros(0)
    all_targets = torch.cat(all_targets, dim=0) if len(all_targets) > 0 else torch.zeros(0)
    r2 = calculate_r2(all_preds, all_targets) if all_preds.numel() else -1.0

    return avg_loss, r2

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
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

    avg_loss = total_loss / max(1, len(loader))
    all_preds = torch.cat(all_preds, dim=0) if len(all_preds) > 0 else torch.zeros(0)
    all_targets = torch.cat(all_targets, dim=0) if len(all_targets) > 0 else torch.zeros(0)
    r2 = calculate_r2(all_preds, all_targets) if all_preds.numel() else -1.0

    return avg_loss, r2

# ---------------------------
# Hyperparameter sampling / search space builder (kept for compatibility)
# ---------------------------
def sample_hyperparameters():
    """Return a sampled config (kept for backward compatibility but not used in exhaustive mode)."""
    # Example sample - you can keep or change this as you like
    config = {
        'lookback': 50,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'bidirectional': False,
        'use_gru': True,
        'fc_num_layers': 2,
        'fc_hidden_dims': 256,
        'fc_activation': 'relu',
        'fc_dropout': 0.0,
        'use_batch_norm': True,
        'batch_size': 256,
        'lr': 0.0001,
        'weight_decay': 0.0,
        'grad_clip': 0.5,
        'optimizer': 'adam',
        'lr_patience': 3,
    }
    return config

def build_search_space():
    """Build a Cartesian product of search params (kept for backward compatibility)."""
    search_space = {
        'lookback': [20, 30, 50],
        'hidden_size': [64, 128, 192],
        'num_layers': [1, 2, 4],
        'dropout': [0.1, 0.2, 0.3],
        'bidirectional': [False, True],
        'use_gru': [True, False],
        'fc_num_layers': [2, 3],
        'fc_hidden_dims': [64, 128, 256],
        'fc_activation': ['relu', 'tanh', 'gelu'],
        'fc_dropout': [0.0, 0.2, 0.3, 0.4],
        'use_batch_norm': [True, False],
        'batch_size': [128, 256],
        'lr': [1e-4, 5e-4, 1e-3],
        'weight_decay': [0.0, 1e-4, 1e-3, 1e-2],
        'grad_clip': [None, 0.5, 1.0, 5.0],
        'optimizer': ['adam', 'adamw'],
        'lr_patience': [3],
    }

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    configs = []
    for prod in itertools.product(*values):
        cfg = dict(zip(keys, prod))
        if cfg['num_layers'] == 1:
            cfg['dropout'] = 0.0
        cfg['batch_size'] = int(cfg['batch_size'])
        configs.append(cfg)
    return configs

# ---------------------------
# Device detection
# ---------------------------
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

# ---------------------------
# Train model wrapper (uses create_model_from_config_new)
# ---------------------------
def train_model(config, train_df, val_df, feature_cols, device, max_epochs=20, patience=5, logger=None):
    n_features = len(feature_cols)

    train_dataset = SequenceDataset(train_df, config['lookback'], feature_cols)
    val_dataset = SequenceDataset(val_df, config['lookback'], feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Map user config to new factory expected keys (provide sensible defaults and preserve overrides)
    factory_config = dict(config)  # shallow copy
    # Encoder hidden: prefer explicit override, otherwise use legacy hidden_size
    factory_config['encoder_hidden'] = int(factory_config.get('encoder_hidden', factory_config.get('hidden_size', 128)))
    # Decoder hidden: default to half of encoder (but at least 64)
    factory_config['decoder_hidden'] = int(factory_config.get('decoder_hidden', max(64, factory_config['encoder_hidden'] // 2)))
    # Encoder / decoder layer counts
    factory_config['encoder_num_layers'] = int(factory_config.get('encoder_num_layers', factory_config.get('num_layers', 1)))
    factory_config['decoder_num_layers'] = int(factory_config.get('decoder_num_layers', factory_config.get('decoder_num_layers', 1)))
    # Dropconnect key name compatibility
    factory_config['dropconnect'] = float(factory_config.get('dropconnect', factory_config.get('dropconnect_p', 0.1)))
    # Attention heads / attn_dim defaults preserved
    factory_config['attention_heads'] = int(factory_config.get('attention_heads', factory_config.get('attn_heads', 4)))
    # Keep dropout, use_gru, use_batch_norm keys as-is

    # Create model via new factory; decoder/encoder input dims are implicitly n_features
    model = create_model_from_config_new(factory_config, n_features).to(device)
    print(model)

    # Optimizer selection
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config.get('lr_patience', 3))
    criterion = nn.L1Loss()

    best_val_r2 = -float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    history = []

    for epoch in range(max_epochs):
        train_loss, train_r2 = train_epoch(model, train_loader, criterion, optimizer, device, config.get('grad_clip'))
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
            if logger: logger.info(f"    Early stopped at epoch {epoch+1}")
            break

    # post-train metrics
    if best_epoch > 0:
        best_epoch_data = history[best_epoch - 1]
        overfitting_gap = best_epoch_data['train_r2'] - best_epoch_data['val_r2']
        best_train_r2 = best_epoch_data['train_r2']
    else:
        # fallback in edge case
        best_epoch_data = history[-1] if history else {'train_r2': -1.0, 'val_r2': -1.0}
        overfitting_gap = best_epoch_data['train_r2'] - best_epoch_data['val_r2']
        best_train_r2 = best_epoch_data['train_r2']

    return {
        'best_val_r2': best_val_r2,
        'best_train_r2': best_train_r2,
        'overfitting_gap': overfitting_gap,
        'best_epoch': best_epoch,
        'total_epochs': len(history),
        'n_parameters': model.count_parameters() if hasattr(model, 'count_parameters') else sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_model_state': best_model_state,
        'history': history,
        'factory_config_used': factory_config
    }

# ---------------------------
# Submission creation
# ---------------------------
def create_submission(config, result, feature_cols, config_id, timestamp, logger):
    r2_str = f"{result['best_val_r2']:.4f}".replace('.', '')
    submission_name = f"bruteforce_r2_{r2_str}_{timestamp}_cfg{config_id}"
    submission_dir = Path(f"../submissions/{submission_name}")
    submission_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"🏆 CREATING SUBMISSION - NEW BEST MODEL!")
    logger.info(f"{'='*70}")
    logger.info(f"  Val R²: {result['best_val_r2']:.6f}")
    logger.info(f"  Directory: {submission_dir}")

    # Save checkpoint
    model_filename = "model_best.pt"
    model_path = submission_dir / model_filename
    checkpoint = {
        'model_state_dict': result['best_model_state'],
        'config': result['factory_config_used'],
        'val_r2': result['best_val_r2'],
        'train_r2': result['best_train_r2'],
        'best_epoch': result['best_epoch'],
        'n_parameters': result['n_parameters'],
        'feature_cols': feature_cols,
        'timestamp': timestamp
    }
    torch.save(checkpoint, model_path)
    logger.info(f"  Saved model: {model_filename}")

    # Create solution.py (keeps compatibility with competition API)
    n_features = len(feature_cols)
    lookback = config['lookback']

    # Try to pull encoder/decoder info from factory config if present
    factory_cfg = result.get('factory_config_used', {})
    enc_hidden = factory_cfg.get('encoder_hidden', factory_cfg.get('hidden_size', config.get('hidden_size', 128)))
    dec_hidden = factory_cfg.get('decoder_hidden', max(64, int(enc_hidden)//2))
    enc_layers = factory_cfg.get('encoder_num_layers', factory_cfg.get('num_layers', 1))
    dec_layers = factory_cfg.get('decoder_num_layers', 1)
    dropconnect = factory_cfg.get('dropconnect', factory_cfg.get('dropconnect_p', 0.1))
    attn_heads = factory_cfg.get('attention_heads', factory_cfg.get('attn_heads', 4))
    attn_dim = factory_cfg.get('attn_dim', None)

    model_type = "GRU" if config.get('use_gru') else "LSTM"
    direction = "Bidirectional" if config.get('bidirectional') else "Unidirectional"

    solution_code = f'''"""
{model_type} Encoder-Decoder Solution - Brute Force Search
Config ID: {config_id}
Timestamp: {timestamp}

Best validation R²: {result['best_val_r2']:.6f}
Training R²: {result['best_train_r2']:.6f}
Overfitting gap: {result['overfitting_gap']:.6f}

Architecture (factory):
- Encoder hidden: {enc_hidden}
- Decoder hidden: {dec_hidden}
- Encoder layers: {enc_layers}
- Decoder layers: {dec_layers}
- Bidirectional encoder: {config.get('bidirectional')}
- DropConnect p: {dropconnect}
- Attention heads: {attn_heads}, attn_dim: {attn_dim}

Lookback: {lookback}
Parameters: {result['n_parameters']:,}

Training:
- Optimizer: {config.get('optimizer', 'adam').upper()}
- Learning rate: {config.get('lr')}
- Weight decay: {config.get('weight_decay')}
- Batch size: {config.get('batch_size')}
- Best epoch: {result['best_epoch']}/{result['total_epochs']}
"""

import numpy as np
import torch
import torch.nn as nn
from utils import DataPoint

# NOTE: This submission loads the saved checkpoint and expects
# the competition runtime to supply DataPoint objects as used before.

class PredictionModel:
    def __init__(self, lookback={lookback}, n_features={n_features}):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')

        # We instantiate a minimal placeholder model whose architecture
        # mirrors training-time factory. For the submission the actual
        # model weights are loaded from model_best.pt saved alongside.
        # For deterministic simplicity the submission uses a small wrapper.

        # Lazy load checkpoint then build a model architecture if needed.
        ckpt = torch.load('{model_filename}', map_location=self.device)
        # We do not attempt to replicate the full factory here; instead,
        # for inference we load the saved state dict into a simple compatible container.
        # A minimal model definition consistent with the saved state is required here.
        # For brevity, the submission uses the same design as during training:
        try:
            # If the saved checkpoint contains a full 'config' we can reconstruct the model
            cfg = ckpt.get('config', {{}})
            from models import create_model_from_config_new
            model = create_model_from_config_new(cfg, n_features)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            self.model = model.to(self.device)
        except Exception as e:
            # Fallback: use simple average predictions if reconstruction fails
            self.model = None
            print("WARNING: could not reconstruct model in submission:", e)

        self.current_seq_ix = None
        self.sequence_history = []

    def predict(self, data_point: DataPoint):
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        self.sequence_history.append(data_point.state.copy())

        if not data_point.need_prediction:
            return None

        if len(self.sequence_history) < self.lookback or self.model is None:
            return np.mean(self.sequence_history, axis=0)

        sequence = np.array(self.sequence_history[-self.lookback:], dtype=np.float32)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(sequence_tensor)
            return out.cpu().numpy()[0]
'''

    solution_path = submission_dir / 'solution.py'
    with open(solution_path, 'w') as f:
        f.write(solution_code)
    logger.info(f"  Generated: solution.py")

    # copy utils.py if available
    utils_src = Path('../competition_package/utils.py')
    utils_dst = submission_dir / 'utils.py'
    if utils_src.exists():
        shutil.copy(utils_src, utils_dst)
        logger.info(f"  Copied: utils.py")
    else:
        logger.info("  models.py not found in ../competition_package; skipping copy.")
    models_src = Path('models.py')
    models_des = submission_dir / 'models.py'
    if models_src.exists():
        shutil.copy(models_src, models_des)
        logger.info(f"  Copied: models.py")
    else:
        logger.info("  models.py not found in root; skipping copy.")

    # Save metadata
        # ----- BUILD COMPLETE METADATA -----
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
    logger.info("=" * 70 + "\n")

    return str(submission_dir)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Brute force hyperparameter search with auto-submission')
    parser.add_argument('--test', action='store_true', help='Test mode (tiny data, 3 configs)')
    parser.add_argument('--n_configs', type=int, default=100, help='Number of configurations to test (cap for exhaustive mode)')
    parser.add_argument('--max_epochs', type=int, default=30, help='Max epochs per config')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'mps', 'cuda', 'auto'], help='Device to use for training (default: auto-detect)')
    parser.add_argument('--best_r2', type=float, default=0.34, help='Starting best R² to beat (default: 0.34)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the exhaustive config order (deterministic seed is used)')
    parser.add_argument('--max_configs', type=int, default=None, help='Maximum number of configurations to run from the full Cartesian product (overrides --n_configs if set)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger, log_file = setup_logging(timestamp)

    logger.info("="*70)
    logger.info("BRUTE FORCE HYPERPARAMETER SEARCH (ENC-DEC LSTM)")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {'TEST' if args.test else 'FULL'}")
    logger.info(f"Best R² to beat: {args.best_r2:.6f}")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    device_str, has_mps, has_cuda = detect_device(args.device)
    device = torch.device(device_str)

    logger.info("Device Detection:")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  Requested: {args.device}")
    logger.info(f"  MPS available: {'✓' if has_mps else '✗'}")
    logger.info(f"  CUDA available: {'✓' if has_cuda else '✗'}")
    logger.info(f"  Using: {device_str.upper()}")
    logger.info("")

    # Seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    logger.info("Loading data...")
    if args.test:
        train_df = pd.read_parquet('data/tiny_train.parquet')
        val_df = pd.read_parquet('data/tiny_val.parquet')
        effective_max_configs = 1000
        effective_max_epochs = 30
        output_file = 'models/test_bruteforce_search_results.csv'
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
        output_file = f'models/bruteforce_search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    feature_cols = [col for col in train_df.columns if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]

    logger.info(f"Train: {len(train_df):,} rows, {train_df['seq_ix'].nunique()} sequences")
    logger.info(f"Val:   {len(val_df):,} rows, {val_df['seq_ix'].nunique()} sequences")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info("")

    Path('models').mkdir(exist_ok=True)
    Path('../submissions').mkdir(exist_ok=True)

    best_global_r2 = args.best_r2
    submissions_log = []

    # Build exhaustive search space (adjust these lists to tune search size)
    search_space = {
        'lookback': [50],
        'hidden_size': [128, 192],
        'num_layers': [1, 2],
        'dropout': [0.1, 0.3],
        'bidirectional': [True],
        'use_gru': [False],
        'fc_num_layers': [1, 2],
        'fc_hidden_dims': [128, 192],
        'fc_activation': ['relu', 'gelu'],
        'fc_dropout': [0.1, 0.3],
        'use_batch_norm': [True, False],
        'batch_size': [128],
        'lr': [1e-4, 5e-4],
        'weight_decay': [1e-4],
        'grad_clip': [5.0],
        'optimizer': ['adam', 'adamw'],
        'lr_patience': [3],
    }

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    logger.info(f"Total possible combinations (Cartesian product): {total_combinations:,}")

    # Determine how many configs to run
    if args.test:
        max_to_run = effective_max_configs
    else:
        if args.max_configs is not None:
            max_to_run = min(args.max_configs, total_combinations)
        else:
            max_to_run = min(effective_max_configs, total_combinations)
    logger.info(f"Configurations to run (cap): {max_to_run}")

    product_iter = itertools.product(*values)

    # Controlled shuffle if requested
    if args.shuffle:
        logger.info("Shuffling configuration order (deterministic seed).")
        limited = list(itertools.islice(product_iter, max_to_run))
        random.seed(42)
        random.shuffle(limited)
        configs_iterable = (dict(zip(keys, tup)) for tup in limited)
    else:
        configs_iterable = (dict(zip(keys, tup)) for tup in itertools.islice(product_iter, max_to_run))

    # Run search
    results = []
    start_time = time.time()

    logger.info("Starting exhaustive brute force search...\n")

    for i, config in enumerate(tqdm(configs_iterable, total=max_to_run, desc="Searching"), start=1):
        config_start = time.time()

        # Normalize types
        config['batch_size'] = int(config.get('batch_size', 256))
        config['lr_patience'] = int(config.get('lr_patience', 3))

        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Config {i}/{max_to_run} - Starting Training:")
            logger.info(f"  Architecture:")
            logger.info(f"    Lookback: {config['lookback']}, Hidden: {config.get('hidden_size')}, "
                        f"Layers: {config.get('num_layers')}, Dropout: {config.get('dropout')}")
            logger.info(f"    Bidirectional: {config.get('bidirectional')}, GRU: {config.get('use_gru')}")
            logger.info(f"  Output Head:")
            logger.info(f"    FC Layers: {config.get('fc_num_layers')}, FC Hidden: {config.get('fc_hidden_dims')}, "
                        f"Activation: {config.get('fc_activation')}")
            logger.info(f"    FC Dropout: {config.get('fc_dropout')}, Batch Norm: {config.get('use_batch_norm')}")
            logger.info(f"  Training:")
            logger.info(f"    Batch Size: {config['batch_size']}, LR: {config['lr']}, "
                        f"Weight Decay: {config['weight_decay']}")
            logger.info(f"    Optimizer: {config['optimizer']}, Grad Clip: {config['grad_clip']}, "
                        f"LR Patience: {config['lr_patience']}")

            # Train model
            result = train_model(
                config, train_df, val_df, feature_cols, device,
                max_epochs=effective_max_epochs, patience=args.patience, logger=logger
            )

            full_result = {**config, **result}
            full_result['config_id'] = i
            full_result['training_time_sec'] = time.time() - config_start
            full_result_csv = {k: v for k, v in full_result.items() if k != 'best_model_state'}
            results.append(full_result_csv)

            logger.info(f"\n{'='*70}")
            logger.info(f"Config {i}/{max_to_run} Complete:")
            logger.info(f"  Best Val R²: {result['best_val_r2']:.6f} (epoch {result['best_epoch']})")
            logger.info(f"  Best Global R²: {best_global_r2:.6f}")
            logger.info(f"  Overfitting Gap: {result['overfitting_gap']:.6f}")
            logger.info(f"  Parameters: {result['n_parameters']:,}")
            logger.info(f"  Training Time: {full_result['training_time_sec']:.1f}s")

            # Auto-submission if improved
            if result['best_val_r2'] > best_global_r2:
                logger.info(f"\n🎉🎉🎉 NEW GLOBAL BEST! R² improved from {best_global_r2:.6f} to {result['best_val_r2']:.6f}")
                improvement = result['best_val_r2'] - best_global_r2
                best_global_r2 = result['best_val_r2']

                submission_dir = create_submission(config, result, feature_cols, i, timestamp, logger)

                submissions_log.append({
                    'config_id': i,
                    'val_r2': result['best_val_r2'],
                    'improvement': improvement,
                    'submission_dir': submission_dir,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

                submissions_df = pd.DataFrame(submissions_log)
                submissions_df.to_csv('models/bruteforce_submissions_log.csv', index=False)
            else:
                logger.info(f"  Not better than global best ({best_global_r2:.6f})")

            # Save results after every config
            results_df = pd.DataFrame(results)
            results_df_sorted = results_df.sort_values('best_val_r2', ascending=False)
            results_df_sorted.to_csv(output_file, index=False)
            logger.info(f"  Results saved to {output_file}")

        except Exception as e:
            logger.info(f"\nERROR in config {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final save
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('best_val_r2', ascending=False)
        results_df.to_csv(output_file, index=False)

    elapsed = time.time() - start_time

    logger.info(f"\n{'='*70}")
    logger.info("SEARCH COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Successful configs: {len(results)}/{max_to_run}")
    logger.info(f"Final best R²: {best_global_r2:.6f}")
    logger.info(f"Submissions created: {len(submissions_log)}")
    logger.info(f"Results saved: {output_file}")
    logger.info("")

    if not results_df.empty:
        logger.info("Top 5 configurations:")
        logger.info("\n" + results_df.head(5)[['config_id', 'best_val_r2', 'overfitting_gap',
                                   'lookback', 'hidden_size', 'num_layers',
                                   'dropout', 'n_parameters']].to_string(index=False))

    if len(submissions_log) > 0:
        logger.info(f"\nSubmissions log saved: models/bruteforce_submissions_log.csv")
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
