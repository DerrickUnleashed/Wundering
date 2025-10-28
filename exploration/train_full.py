#!/usr/bin/env python3
"""
Retrain final model on 100% of data using the chosen best configuration,
and generate a submission folder (solution.py + model_best.pt + utils.py + metadata.json).

Usage:
    python3 train_final_best.py --max_epochs 100 --batch_size 256 --output_root ../submissions
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime
import json
import argparse
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from tqdm import tqdm

# Ensure local modules are importable (assumes repo structure same as prior script)
sys.path.append('.')
sys.path.append('../competition_package')
from models import create_model_from_config  # same factory used earlier

# --------- Helper dataset & utils (copied/adapted from your brute-force script) ----------
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
    """Calculate mean R² across features."""
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()

    r2_scores = []
    # protect against degenerate shapes
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    for i in range(predictions.shape[1]):
        # if constant target, r2_score may warn; handle gracefully
        try:
            r2 = r2_score(targets[:, i], predictions[:, i])
        except Exception:
            r2 = float('nan')
        r2_scores.append(r2)
    # return mean ignoring nan
    r2_arr = np.array(r2_scores, dtype=np.float32)
    return float(np.nanmean(r2_arr))


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
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = calculate_r2(all_preds, all_targets)
    return avg_loss, r2


def evaluate(model, loader, criterion, device):
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
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = calculate_r2(all_preds, all_targets)
    return avg_loss, r2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------ Fixed best configuration (as requested) --------------------
BEST_CONFIG = {
    # Architecture
    'lookback': 50,
    'hidden_size': 128,
    'num_layers': 1,
    'dropout': 0.1,
    'bidirectional': True,
    'use_gru': True,

    # Output head
    'fc_num_layers': 2,
    'fc_hidden_dims': 256,
    'fc_activation': 'relu',
    'fc_dropout': 0.2,
    'use_batch_norm': True,

    # Training (defaults; can be overridden via CLI)
    'batch_size': 256,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'grad_clip': 5.0,
    'optimizer': 'adam',
    'lr_patience': 3,
}

# ------------------ Script entrypoint --------------------
def main():
    parser = argparse.ArgumentParser(description="Retrain final model on full dataset and create submission.")
    parser.add_argument('--max_epochs', type=int, default=60, help='Number of epochs for final training.')
    parser.add_argument('--batch_size', type=int, default=BEST_CONFIG['batch_size'], help='Batch size (overrides config).')
    parser.add_argument('--lr', type=float, default=BEST_CONFIG['lr'], help='Learning rate (overrides config).')
    parser.add_argument('--weight_decay', type=float, default=BEST_CONFIG['weight_decay'], help='Weight decay.')
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','mps','cuda'], help='Device to use.')
    parser.add_argument('--output_root', type=str, default='../submissions', help='Where to place submission dir.')
    parser.add_argument('--timestamp', type=str, default=None, help='Optional timestamp for submission folder naming.')
    args = parser.parse_args()

    # device detection (same logic as before)
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    has_cuda = torch.cuda.is_available()
    if args.device == 'auto':
        if has_mps:
            device_str = 'mps'
        elif has_cuda:
            device_str = 'cuda'
        else:
            device_str = 'cpu'
    else:
        device_str = args.device if args.device in ('mps','cuda','cpu') else 'cpu'
    device = torch.device(device_str)

    # set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # load full data (assumes same path as competition package used previously)
    data_path = Path('../competition_package/datasets/train.parquet')
    if not data_path.exists():
        raise FileNotFoundError(f"Expected dataset at {data_path.resolve()} - update path or place dataset there.")
    full_df = pd.read_parquet(str(data_path))

    # Use all sequences for training (we will still compute training R² for diagnostics)
    train_df = full_df.copy()
    feature_cols = [c for c in train_df.columns if c not in ['seq_ix', 'step_in_seq', 'need_prediction']]

    lookback = BEST_CONFIG['lookback']
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    max_epochs = int(args.max_epochs)

    print(f"Device: {device_str}, Using full dataset: {len(train_df):,} rows, {train_df['seq_ix'].nunique()} sequences")
    print(f"Features: {len(feature_cols)}, Lookback: {lookback}")
    print("Using configuration (overrides shown):")
    cfg = BEST_CONFIG.copy()
    cfg['batch_size'] = batch_size
    cfg['lr'] = lr
    cfg['weight_decay'] = weight_decay
    print(json.dumps(cfg, indent=2))

    # build dataset & loader (train loader used for full training; we will use it also for evaluation)
    train_dataset = SequenceDataset(train_df, lookback, feature_cols)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # create model
    model = create_model_from_config(cfg, len(feature_cols)).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # optimizer
    if cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler: optional but useful; reduce when train R² plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=max(5, cfg.get('lr_patience', 3)), min_lr=1e-6)

    criterion = nn.MSELoss()

    best_train_r2 = -float('inf')
    best_epoch = 0
    best_state = None
    history = []

    # training loop
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss, train_r2 = train_epoch(model, train_loader, criterion, optimizer, device, cfg.get('grad_clip'))
        # evaluate on training set (we don't have a val set here)
        eval_loss, eval_r2 = evaluate(model, train_loader, criterion, device)

        scheduler.step(eval_r2)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_r2': train_r2,
            'eval_loss': eval_loss,
            'eval_r2': eval_r2,
            'lr': optimizer.param_groups[0]['lr']
        })

        marker = "✓" if eval_r2 > best_train_r2 else " "

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Epoch {epoch:3d}/{max_epochs}: Train R²={train_r2:.6f} Eval(Retrain) R²={eval_r2:.6f} LR={optimizer.param_groups[0]['lr']:.6g} {marker}")

        if eval_r2 > best_train_r2 + 1e-8:
            best_train_r2 = eval_r2
            best_epoch = epoch
            best_state = model.state_dict()
            # save interim best
            tmp_dir = Path('models')
            tmp_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state,
                'train_r2': best_train_r2,
                'config': cfg
            }, tmp_dir / 'final_best_tmp.pt')

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed/60:.1f} minutes. Best train R²: {best_train_r2:.6f} at epoch {best_epoch}")

    # Prepare submission directory
    ts = args.timestamp if args.timestamp else datetime.now().strftime('%Y%m%d_%H%M%S')
    r2_str = f"{best_train_r2:.4f}".replace('.', '')
    submission_name = f"bruteforce_r2_{r2_str}_{ts}_final"
    submission_dir = Path(args.output_root) / submission_name
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint (same keys as earlier submission format)
    model_filename = 'model_best.pt'
    checkpoint = {
        'model_state_dict': best_state if best_state is not None else model.state_dict(),
        'config': cfg,
        'val_r2': float(best_train_r2),   # no val set; store train R² here for traceability
        'train_r2': float(best_train_r2),
        'best_epoch': best_epoch,
        'n_parameters': n_params,
        'feature_cols': feature_cols,
        'timestamp': ts
    }
    torch.save(checkpoint, submission_dir / model_filename)
    print(f"Saved model to {submission_dir / model_filename}")

    # Generate solution.py (same style as earlier script)
    n_features = len(feature_cols)
    lookback = cfg['lookback']
    model_type = "GRU" if cfg['use_gru'] else "LSTM"
    direction = "Bidirectional" if cfg['bidirectional'] else "Unidirectional"

    solution_code = f'''"""
{model_type} Final Solution - Retrained on Full Data
Timestamp: {ts}

Trained (retrain) R²: {best_train_r2:.6f}
Architecture:
- {model_type} ({direction})
- {cfg['num_layers']} layers, {cfg['hidden_size']} hidden units
- Lookback: {lookback} timesteps
- Dropout: {cfg['dropout']}
- FC layers: {cfg['fc_num_layers']}, activation: {cfg['fc_activation']}
- Parameters: {n_params:,}

Training:
- Optimizer: {cfg['optimizer'].upper()}
- Learning rate: {cfg['lr']}
- Weight decay: {cfg['weight_decay']}
- Batch size: {cfg['batch_size']}
- Best epoch: {best_epoch}/{len(history)}
"""

import numpy as np
import torch
import torch.nn as nn
from utils import DataPoint


class FlexibleRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size={cfg['hidden_size']},
        num_layers={cfg['num_layers']},
        dropout={cfg['dropout']},
        bidirectional={cfg['bidirectional']},
        use_gru={cfg['use_gru']},
        fc_num_layers={cfg['fc_num_layers']},
        fc_hidden_dims={cfg['fc_hidden_dims']},
        fc_activation='{cfg['fc_activation']}',
        fc_dropout={cfg['fc_dropout']},
        use_batch_norm={cfg['use_batch_norm']}
    ):
        super(FlexibleRNN, self).__init__()
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        rnn_out_size = hidden_size * (2 if bidirectional else 1)
        self.output_head = self._build_output_head(
            rnn_out_size, input_size, fc_num_layers,
            fc_hidden_dims, fc_activation, fc_dropout, use_batch_norm
        )

    def _build_output_head(self, input_dim, output_dim, num_layers,
                           hidden_dims, activation, dropout, use_batch_norm):
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
    def __init__(self, lookback={lookback}, n_features={n_features}):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')
        self.model = FlexibleRNN(input_size=n_features).to(self.device)
        checkpoint = torch.load('{model_filename}', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.current_seq_ix = None
        self.sequence_history = []

    def predict(self, data_point: DataPoint):
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []
        self.sequence_history.append(data_point.state.copy())
        if not data_point.need_prediction:
            return None
        if len(self.sequence_history) < self.lookback:
            return np.mean(self.sequence_history, axis=0)
        sequence = np.array(self.sequence_history[-self.lookback:], dtype=np.float32)
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
    print(f"Wrote solution.py to {solution_path}")

    # Copy utils.py (must exist)
    utils_src = Path('../competition_package/utils.py')
    if not utils_src.exists():
        print(f"Warning: expected utils.py at {utils_src.resolve()}; please copy manually.")
    else:
        shutil.copy(utils_src, submission_dir / 'utils.py')
        print(f"Copied utils.py to {submission_dir / 'utils.py'}")

    # Save metadata.json
    metadata = {
        'config_id': 'final_retrain',
        'timestamp': ts,
        'val_r2': float(best_train_r2),
        'train_r2': float(best_train_r2),
        'overfitting_gap': None,
        'best_epoch': best_epoch,
        'total_epochs': len(history),
        'n_parameters': n_params,
        'config': cfg
    }
    with open(submission_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata.json to {submission_dir / 'metadata.json'}")

    print(f"Final submission directory created: {submission_dir.resolve()}")
    print("Done.")


if __name__ == '__main__':
    main()
