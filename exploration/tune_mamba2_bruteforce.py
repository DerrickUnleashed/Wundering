#!/usr/bin/env python3
"""
Brute Force Hyperparameter Search with Auto-Submission (Mamba2-style)

This file implements a self-contained, simplified PyTorch Mamba2-like block
with a cache/step API for single-step inference, plus a brute-force
hyperparameter search scaffold (training/validation, result logging,
and automatic submission creation).

Notes:
- This is a simplified, pure-PyTorch implementation inspired by
  the mamba2_simple reference. It does not depend on Triton ops and
  consciously simplifies the SSM internals to be readable and runnable.
- The model supports a forward path for sequences and a step() method
  for single-step autoregressive decoding that uses a tiny conv cache.
"""

import sys
import os
import math
import json
import itertools
import random
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from tqdm import tqdm
import shutil
import logging
import argparse
import torch.optim as optim

# Force unbuffered IO
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Add local package path (competition_package contains utils/DataPoint etc.)
sys.path.append('.')
sys.path.append('../competition_package')

# ----------------------------- Config ---------------------------------
@dataclass
class Mamba2Config:
    d_model: int
    n_layers: int = 2
    d_head: int = 16
    d_state: int = 64
    expand_factor: int = 2
    d_conv: int = 4
    n_groups: int = 1
    learnable_init_states: bool = False
    activation: str = 'silu'
    rms_norm_eps: float = 1e-5
    bias: bool = False
    conv_bias: bool = True
    mup: bool = False
    chunk_size: int = 256
    dtype = None
    device = None

    def __post_init__(self):
        self.d_inner = int(self.expand_factor * self.d_model)
        self.n_heads = self.d_inner // self.d_head
        assert self.d_inner % self.d_head == 0


# -------------------------- Simple RMSNorm ------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()
        self.use_mup = use_mup
        self.eps = eps
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))
        else:
            # muP-backed behavior: fixed gain
            self.register_buffer('weight', torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        out = x / rms
        return out * self.weight


# -------------------------- Mamba2Block (simplified) --------------------
class Mamba2Block(nn.Module):
    """Simplified Mamba2-like block with cache/step API.

    - forward(x, cache=None): parallel processing for sequences
    - step(x_step, conv_cache): single-step update returning (out_step, new_conv_cache)
    """
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg

        # Input projection splits into conv branch and small dt/head branch
        conv_dim = cfg.d_inner + 2 * cfg.n_groups * cfg.d_state
        self.in_proj = nn.Linear(cfg.d_model, conv_dim + cfg.n_heads, bias=cfg.bias)

        # Depthwise conv across time (groups=conv_dim -> depthwise)
        self.conv1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=cfg.d_conv,
                                padding=cfg.d_conv - 1, groups=conv_dim, bias=cfg.conv_bias)

        # Norm and out projection
        self.norm = RMSNorm(cfg.d_inner, eps=cfg.rms_norm_eps, use_mup=cfg.mup)
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=cfg.bias)
        self.activation = F.silu if cfg.activation in ('silu', 'swish') else F.gelu

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B, L, D)
        B, L, D = x.shape

        # If cache passed and L == 1, use step path
        if cache is not None and L == 1:
            out_step, new_cache = self.step(x, cache)
            return out_step, new_cache

        # Parallel path
        z = self.in_proj(x)  # (B, L, conv_dim + n_heads)
        conv_dim = self.cfg.d_inner + 2 * self.cfg.n_groups * self.cfg.d_state
        xBC = z[..., :conv_dim]
        # dt/head part ignored in simplified implementation -- kept for compatibility

        # conv1d expects (B, C, L)
        xBC_conv = self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)  # (B, L, conv_dim)
        xBC_conv = self.activation(xBC_conv)
        # conv1d with padding may extend the time dimension by (d_conv-1).
        # Trim the conv output to match the original sequence length so the
        # residual addition (out + x) uses matching shapes.
        if xBC_conv.shape[1] != L:
            # take the last L timesteps (causal alignment)
            xBC_conv = xBC_conv[:, -L:, :]

        # split into x, B, C
        x_part, Bpart, Cpart = torch.split(xBC_conv, [self.cfg.d_inner, self.cfg.n_groups * self.cfg.d_state, self.cfg.n_groups * self.cfg.d_state], dim=-1)

        # apply RMSNorm on x_part (per time-step) and project
        y = self.norm(x_part)
        out = self.out_proj(y)

        # we don't maintain a complicated cache for training forward; return None cache
        return out, None

    def step(self, u: torch.Tensor, conv_cache: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # u: (B, 1, D) -> we return (B, 1, D) and updated conv_cache
        B, L, D = u.shape
        assert L == 1

        # in_proj for single step
        z = self.in_proj(u.squeeze(1))  # (B, conv_dim + n_heads)
        conv_dim = self.cfg.d_inner + 2 * self.cfg.n_groups * self.cfg.d_state
        xBC = z[..., :conv_dim]

        # initialize conv_cache if needed: shape (B, conv_dim, d_conv-1)
        if conv_cache is None:
            conv_cache = torch.zeros(B, conv_dim, max(0, self.cfg.d_conv - 1), device=xBC.device, dtype=xBC.dtype)

        if self.cfg.d_conv > 1:
            # roll and append new xBC
            conv_cache = torch.roll(conv_cache, shifts=-1, dims=-1)
            conv_cache[:, :, -1] = xBC
            # compute conv with kernel weights
            # conv1d.weight shape: (conv_dim, 1, k)
            weight = self.conv1d.weight.view(conv_dim, -1)  # (conv_dim, k)
            x_pieces = conv_cache * weight.unsqueeze(0)  # (B, conv_dim, k)
            xBC_conv = x_pieces.sum(dim=-1)
            if self.conv1d.bias is not None:
                xBC_conv = xBC_conv + self.conv1d.bias
            xBC_conv = self.activation(xBC_conv)
        else:
            xBC_conv = self.activation(xBC)

        x_part, Bpart, Cpart = torch.split(xBC_conv, [self.cfg.d_inner, self.cfg.n_groups * self.cfg.d_state, self.cfg.n_groups * self.cfg.d_state], dim=-1)

        y = self.norm(x_part)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_cache


# -------------------------- Residual stack & model ----------------------
class ResidualBlock(nn.Module):
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps, use_mup=cfg.mup)
        self.mixer = Mamba2Block(cfg)

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        # apply norm -> mixer -> residual
        nx = self.norm(x)
        out, new_cache = self.mixer(nx, cache)
        return out + x, new_cache

    def get_empty_cache(self, batch_size: int):
        # conv cache shape (B, conv_dim, d_conv-1)
        conv_dim = self.mixer.cfg.d_inner + 2 * self.mixer.cfg.n_groups * self.mixer.cfg.d_state
        if self.mixer.cfg.d_conv > 1:
            conv_cache = torch.zeros(batch_size, conv_dim, self.mixer.cfg.d_conv - 1)
        else:
            conv_cache = torch.zeros(0)
        return conv_cache


class Mamba2(nn.Module):
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([ResidualBlock(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x: torch.Tensor, caches: Optional[list] = None):
        # x: (B, L, D)
        B = x.shape[0]
        if caches is None:
            caches = [None] * len(self.layers)

        out = x
        for i, layer in enumerate(self.layers):
            out, caches[i] = layer(out, caches[i])

        # if any cache is non-None, return both out and caches to allow decoding
        if any(c is not None for c in caches):
            return out, caches
        return out

    def get_empty_caches(self, batch_size: int):
        return [layer.get_empty_cache(batch_size) for layer in self.layers]


# --------------------------- Dataset & helpers --------------------------
class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int, feature_cols: list):
        self.sequences = []
        self.targets = []
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


def calculate_r2(predictions, targets):
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    r2_scores = []
    for i in range(predictions.shape[1]):
        r2_scores.append(r2_score(targets[:, i], predictions[:, i]))
    return float(np.mean(r2_scores))


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(sequences)
        # model returns (out) or (out, caches)
        if isinstance(preds, tuple):
            preds = preds[0]
        # If model returns a sequence (B, L, D), use last time-step for prediction
        if preds.dim() == 3:
            preds = preds[:, -1, :]
        loss = criterion(preds, targets)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        all_preds.append(preds.detach())
        all_targets.append(targets.detach())
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = calculate_r2(all_preds, all_targets)
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
            preds = model(sequences)
            if isinstance(preds, tuple):
                preds = preds[0]
            if preds.dim() == 3:
                preds = preds[:, -1, :]
            loss = criterion(preds, targets)
            total_loss += loss.item()
            all_preds.append(preds)
            all_targets.append(targets)
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = calculate_r2(all_preds, all_targets)
    return avg_loss, r2


def train_model(config, train_df, val_df, feature_cols, device, max_epochs=20, patience=5, logger=None):
    n_features = len(feature_cols)
    train_dataset = SequenceDataset(train_df, config['lookback'], feature_cols)
    val_dataset = SequenceDataset(val_df, config['lookback'], feature_cols)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    cfg = Mamba2Config(d_model=n_features, n_layers=config.get('n_layers', 2), d_head=config.get('d_head', 16),
                       d_state=config.get('d_state', 64), expand_factor=config.get('expand_factor', 2),
                       d_conv=config.get('d_conv', 4), n_groups=config.get('n_groups', 1))

    model = Mamba2(cfg).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config.get('lr_patience', 3))
    criterion = nn.MSELoss()

    best_val_r2 = -float('inf')
    patience_counter = 0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(max_epochs):
        train_loss, train_r2 = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=config.get('grad_clip', None))
        val_loss, val_r2 = validate(model, val_loader, criterion, device)
        scheduler.step(val_r2)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_r2': train_r2, 'val_loss': val_loss, 'val_r2': val_r2, 'lr': optimizer.param_groups[0]['lr']})
        if logger:
            logger.info(f"Epoch {epoch+1}/{max_epochs}: train_r2={train_r2:.4f} val_r2={val_r2:.4f}")
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if logger:
                logger.info(f"Early stopping at epoch {epoch+1}")
            break

    best_data = history[best_epoch - 1]
    overfitting_gap = best_data['train_r2'] - best_data['val_r2']

    return {
        'best_val_r2': best_val_r2,
        'best_train_r2': best_data['train_r2'],
        'overfitting_gap': overfitting_gap,
        'best_epoch': best_epoch,
        'total_epochs': len(history),
        'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_model_state': best_state,
        'history': history
    }


# --------------------------- Submission & Prediction -------------------
class Mamba2PredictionModel:
    """Lightweight wrapper used inside generated submission/solution.py

    It loads a saved `model_best.pt` checkpoint and uses the model's
    step API for single-step prediction while maintaining a conv cache.
    """
    def __init__(self, checkpoint_path: str, lookback: int, feature_cols: list):
        self.lookback = lookback
        self.feature_cols = feature_cols
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        cfg = Mamba2Config(d_model=len(feature_cols))
        self.model = Mamba2(cfg)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        # prepare per-layer conv caches
        self.caches = self.model.get_empty_caches(batch_size=1)
        self.sequence_history = []

    def predict(self, data_point):
        # data_point: expected to have attributes seq_ix, state (np array), need_prediction
        if getattr(self, 'current_seq_ix', None) != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []
            self.caches = self.model.get_empty_caches(batch_size=1)

        self.sequence_history.append(np.array(data_point.state, dtype=np.float32))
        if not data_point.need_prediction:
            return None

        if len(self.sequence_history) < self.lookback:
            return np.mean(self.sequence_history, axis=0)

        # Use step-by-step decoding: feed the last step into each layer's step
        x = torch.FloatTensor(self.sequence_history[-1:]).unsqueeze(0)  # (1, 1, D)
        # iterate layers' step manually: each ResidualBlock uses Mamba2Block.step semantics
        new_caches = []
        out = x
        for i, layer in enumerate(self.model.layers):
            # call underlying block.step
            block = layer.mixer
            cache = self.caches[i]
            out_step, new_cache = block.step(out, cache)
            # apply residual + norm
            out = out + out_step
            new_caches.append(new_cache)

        self.caches = new_caches
        pred = out.squeeze(0).squeeze(0).numpy()
        return pred


# --------------------------- Search & CLI -------------------------------
def setup_logging(timestamp: str):
    Path('logs').mkdir(exist_ok=True)
    logger = logging.getLogger('mamba2_bruteforce')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(f'logs/mamba2_bruteforce_{timestamp}.log')
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def create_submission(config, result, feature_cols, config_id, timestamp, logger):
    r2_str = f"{result['best_val_r2']:.4f}".replace('.', '')
    submission_name = f"mamba2_r2_{r2_str}_{timestamp}_cfg{config_id}"
    submission_dir = Path(f"../submissions/{submission_name}")
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    checkpoint = {
        'model_state_dict': result['best_model_state'],
        'config': config,
        'val_r2': result['best_val_r2']
    }
    model_path = submission_dir / 'model_best.pt'
    torch.save(checkpoint, model_path)
    logger.info(f"Saved model to {model_path}")

    # Create a minimal solution.py that uses the Mamba2PredictionModel
    n_features = len(feature_cols)
    lookback = config['lookback']

    solution_code = f'''"""
Minimal Mamba2 submission wrapper (auto-generated).

This is a minimal placeholder solution that loads metadata from
`model_best.pt` and provides a simple heuristic predictor. Replace
the body with a full model-based predictor if you want the real
inference behavior in the submission.
"""
import numpy as np
import torch
from utils import DataPoint

class PredictionModel:
    def __init__(self, lookback={lookback}, feature_cols={feature_cols}):
        self.lookback = lookback
        self.feature_cols = feature_cols
        self.ckpt = None
        try:
            self.ckpt = torch.load('model_best.pt', map_location='cpu')
        except Exception:
            self.ckpt = None
        self.current_seq_ix = None
        self.history = []

    def predict(self, data_point: DataPoint):
        # Keep a short history per sequence and return a simple heuristic.
        if getattr(self, 'current_seq_ix', None) != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.history = []

        self.history.append(np.array(data_point.state, dtype=np.float32))

        if not getattr(data_point, 'need_prediction', False):
            return None

        if len(self.history) < self.lookback:
            return np.mean(self.history, axis=0)

        # Fallback naive predictor: return the most recent state.
        return self.history[-1]

(submission_dir / 'solution.py').write_text(solution_code)

    # Copy utils
    utils_src = Path('../competition_package/utils.py')
    if utils_src.exists():
        shutil.copy(utils_src, submission_dir / 'utils.py')

    meta = {
        'config': config,
        'val_r2': result['best_val_r2'],
        'train_r2': result['best_train_r2'],
    }
    (submission_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    logger.info(f"Created submission at {submission_dir}")
    return str(submission_dir)'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--n_configs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--best_r2', type=float, default=0.34)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(timestamp)

    # device selection
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    has_cuda = torch.cuda.is_available()
    if args.device == 'auto':
        device_str = 'mps' if has_mps else ('cuda' if has_cuda else 'cpu')
    else:
        device_str = args.device
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # load data
    logger.info("Loading data...")
    if args.test:
        train_df = pd.read_parquet('data/tiny_train.parquet')
        val_df = pd.read_parquet('data/tiny_val.parquet')
    else:
        full_df = pd.read_parquet('../competition_package/datasets/train.parquet')
        all_seqs = full_df['seq_ix'].unique()
        n_train = int(0.9 * len(all_seqs))
        train_seqs = all_seqs[:n_train]
        val_seqs = all_seqs[n_train:]
        train_df = full_df[full_df['seq_ix'].isin(train_seqs)]
        val_df = full_df[full_df['seq_ix'].isin(val_seqs)]

    feature_cols = [c for c in train_df.columns if c not in ('seq_ix', 'step_in_seq', 'need_prediction')]

    # search space (small by default)
    search_space = {
        'lookback': [20, 30],
        'batch_size': [128],
        'lr': [1e-3, 5e-4],
        'weight_decay': [0.0, 1e-4],
        'grad_clip': [None, 1.0],
        'n_layers': [1, 2],
    }
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    total = 1
    for v in values:
        total *= len(v)
    max_to_run = min(args.n_configs, total)

    configs_iterable = (dict(zip(keys, tup)) for tup in itertools.islice(itertools.product(*values), max_to_run))

    best_global_r2 = args.best_r2
    results = []
    submissions_log = []

    for i, cfg in enumerate(tqdm(configs_iterable, total=max_to_run), start=1):
        cfg['batch_size'] = int(cfg['batch_size'])
        try:
            logger.info(f"\n=== Config {i}/{max_to_run}: {cfg}")
            res = train_model(cfg, train_df, val_df, feature_cols, device, max_epochs=args.max_epochs, patience=args.patience, logger=logger)
            full = {**cfg, **res, 'config_id': i}
            results.append({k: v for k, v in full.items() if k != 'best_model_state'})
            logger.info(f"Best val r2: {res['best_val_r2']:.6f} (global best {best_global_r2:.6f})")
            if res['best_val_r2'] > best_global_r2:
                best_global_r2 = res['best_val_r2']
                subdir = create_submission(cfg, res, feature_cols, i, timestamp, logger)
                submissions_log.append({'config_id': i, 'val_r2': res['best_val_r2'], 'dir': subdir})
            # save results intermittently
            Path('models').mkdir(exist_ok=True)
            pd.DataFrame(results).sort_values('best_val_r2', ascending=False).to_csv(f"models/mamba2_search_results_{timestamp}.csv", index=False)
        except Exception as e:
            logger.exception(f"Error in config {i}: {e}")
            continue

    logger.info("Search complete")
    if results:
        logger.info(pd.DataFrame(results).sort_values('best_val_r2', ascending=False).head().to_string())


if __name__ == '__main__':
    main()

