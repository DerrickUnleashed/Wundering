"""
Train an LSTM model with self-attention on top of RNN outputs.

This script trains a model with an attention mechanism after the RNN layer to better capture
important temporal dependencies. It follows similar data loading, training, and submission logic
to the original LSTM bruteforce script for easy comparison.

Usage:
    python train_lstm_with_attention.py --config config1  # Use predefined config
    python train_lstm_with_attention.py --custom          # Specify custom hyperparameters
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score


# ============================================================================
# Predefined Configurations
# ============================================================================

CONFIGS = {
    'config1': {
        # Config tuned for LSTM with attention
        'lookback': 50,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'bidirectional': True,
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
}


# ============================================================================
# Model Architecture
# ============================================================================

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, bidirectional=False):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.d = hidden_size * (2 if bidirectional else 1)

        self.attention = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.Tanh(),
            nn.Linear(self.d, 1)
        )

    def forward(self, rnn_outputs):
        """
        rnn_outputs: Tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            context vector of shape (batch_size, hidden_dim)
        """
        # Calculate attention weights
        attn_weights = self.attention(rnn_outputs)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize over seq_len

        # Weighted sum of rnn outputs
        context = torch.sum(attn_weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context


class LSTMWithAttention(nn.Module):
    """LSTM-based sequence predictor with self-attention."""

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
        use_gru=False,
        fc_num_layers=1,
        fc_hidden_dims=128,
        fc_activation='relu',
        fc_dropout=0.3,
        use_batch_norm=True
    ):
        super(LSTMWithAttention, self).__init__()

        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.attention = SelfAttention(hidden_size, bidirectional)

        self.d = hidden_size * (2 if bidirectional else 1)

        self.output_head = self._build_output_head(
            self.d, input_size, fc_num_layers,
            fc_hidden_dims, fc_activation, fc_dropout, use_batch_norm
        )

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
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim)
        attention_out = self.attention(rnn_out)
        return self.output_head(attention_out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Data Preparation
# ============================================================================

def create_sequences(df, feature_cols, lookback):
    sequences = []
    targets = []

    for seq_ix in df['seq_ix'].unique():
        seq_data = df[df['seq_ix'] == seq_ix][feature_cols].values

        for i in range(lookback, len(seq_data)):
            sequences.append(seq_data[i - lookback:i])
            targets.append(seq_data[i])

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


# ============================================================================
# Training
# ============================================================================

def train_model(config, train_df, val_df, feature_cols, device, max_epochs=40):
    print("\n" + "=" * 70)
    print("TRAINING MODEL WITH ATTENTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Architecture:")
    print(f"    Lookback: {config['lookback']}, Hidden: {config['hidden_size']}, "
          f"Layers: {config['num_layers']}, Dropout: {config['dropout']}")
    print(f"    Bidirectional: {config['bidirectional']}, GRU: {config['use_gru']}")
    print(f"    Self-Attention added")
    print(f"  Output Head:")
    print(f"    FC Layers: {config['fc_num_layers']}, FC Hidden: {config['fc_hidden_dims']}, "
          f"Activation: {config['fc_activation']}")
    print(f"    FC Dropout: {config['fc_dropout']}, Batch Norm: {config['use_batch_norm']}")
    print(f"  Training:")
    print(f"    Batch Size: {config['batch_size']}, LR: {config['lr']}, "
          f"Weight Decay: {config['weight_decay']}")
    print(f"    Optimizer: {config['optimizer']}, Grad Clip: {config['grad_clip']}, "
          f"LR Patience: {config['lr_patience']}")

    print(f"\nPreparing sequences (lookback={config['lookback']})...")
    X_train, y_train = create_sequences(train_df, feature_cols, config['lookback'])
    X_val, y_val = create_sequences(val_df, feature_cols, config['lookback'])

    print(f"  Train sequences: {len(X_train):,}")
    print(f"  Val sequences: {len(X_val):,}")

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    model = LSTMWithAttention(
        input_size=len(feature_cols),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        use_gru=config['use_gru'],
        fc_num_layers=config['fc_num_layers'],
        fc_hidden_dims=config['fc_hidden_dims'],
        fc_activation=config['fc_activation'],
        fc_dropout=config['fc_dropout'],
        use_batch_norm=config['use_batch_norm']
    ).to(device)

    n_params = model.count_parameters()
    print(f"\nModel parameters: {n_params:,}")

    if config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                               weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                               weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                              weight_decay=config['weight_decay'], momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=config['lr_patience']
    )

    criterion = nn.MSELoss()

    print("\nStarting training...")
    print("-" * 70)

    best_val_r2 = -float('inf')
    best_train_r2 = None
    best_epoch = 0
    best_model_state = None
    patience = 5
    patience_counter = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            if config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            optimizer.step()

            train_loss += loss.item()
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(y_batch.cpu().numpy())

        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_r2 = r2_score(train_targets.flatten(), train_preds.flatten())

        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_r2 = r2_score(val_targets.flatten(), val_preds.flatten())

        scheduler.step(val_r2)

        gap = train_r2 - val_r2
        history.append({
            'epoch': epoch + 1,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'gap': gap
        })

        marker = ""
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_train_r2 = train_r2
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            patience_counter = 0
            marker = "✓"
        else:
            patience_counter += 1

        msg = f"    Epoch {epoch+1:2d}/{max_epochs}: Val R²={val_r2:.4f} Train R²={train_r2:.4f} Gap={gap:.4f} {marker}"
        print(msg)

        if patience_counter >= patience:
            print(f"    Early stopped at epoch {epoch+1}")
            break

    print("-" * 70)

    # Fix potential None subtraction issue for overfitting gap
    if best_train_r2 is None or best_val_r2 is None:
        overfitting_gap = None
    else:
        overfitting_gap = best_train_r2 - best_val_r2

    result = {
        'best_val_r2': best_val_r2,
        'best_train_r2': best_train_r2,
        'overfitting_gap': overfitting_gap,
        'best_epoch': best_epoch,
        'total_epochs': len(history),
        'n_parameters': n_params,
        'best_model_state': best_model_state,
        'history': history
    }

    print(f"\nTraining Complete:")
    print(f"  Best Val R²: {best_val_r2:.6f} (epoch {best_epoch})")
    print(f"  Best Train R²: {best_train_r2:.6f}")
    print(f"  Overfitting Gap: {overfitting_gap:.6f}")
    print(f"  Parameters: {n_params:,}")

    return result


# ============================================================================
# Submission Creation
# ============================================================================

def create_submission(config, result, feature_cols, config_name, timestamp):
    r2_str = f"{result['best_val_r2']:.4f}".replace('.', '')
    submission_name = f"lstm_attention_{config_name}_r2_{r2_str}_{timestamp}"
    submission_dir = Path(f"../submissions/{submission_name}")
    submission_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CREATING SUBMISSION")
    print("=" * 70)
    print(f"  Val R²: {result['best_val_r2']:.6f}")
    print(f"  Directory: {submission_dir}")

    model_filename = "model_best.pt"
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
    print(f"  Saved model: {model_filename}")

    model_type = "GRU" if config['use_gru'] else "LSTM"
    direction = "Bidirectional" if config['bidirectional'] else "Unidirectional"
    attention_str = "with Self-Attention"

    solution_code = f'''"""
{model_type} Solution {attention_str} - {config_name}
Timestamp: {timestamp}

Best validation R²: {result['best_val_r2']:.6f}
Training R²: {result['best_train_r2']:.6f}
Overfitting gap: {result['overfitting_gap']:.6f}

Architecture:
- {model_type} ({direction}) {attention_str}
- {config['num_layers']} layers, {config['hidden_size']} hidden units
- Lookback: {config['lookback']} timesteps
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

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, bidirectional=False):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.d = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.Tanh(),
            nn.Linear(self.d, 1)
        )
    def forward(self, rnn_outputs):
        attn_weights = self.attention(rnn_outputs)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * rnn_outputs, dim=1)
        return context

class LSTMWithAttention(nn.Module):
    def __init__(self,
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
                 use_batch_norm={config['use_batch_norm']}):
        super(LSTMWithAttention, self).__init__()
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0,
                             bidirectional=bidirectional)
        self.attention = SelfAttention(hidden_size, bidirectional)
        self.d = hidden_size * (2 if bidirectional else 1)
        self.output_head = self._build_output_head(self.d,
                                                   input_size,
                                                   fc_num_layers,
                                                   fc_hidden_dims,
                                                   fc_activation,
                                                   fc_dropout,
                                                   use_batch_norm)
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
        attention_out = self.attention(rnn_out)
        return self.output_head(attention_out)

class PredictionModel:
    def __init__(self, lookback={config['lookback']}, n_features={len(feature_cols)}):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')
        self.model = LSTMWithAttention(input_size=n_features,
                                      hidden_size={config['hidden_size']},
                                      num_layers={config['num_layers']},
                                      dropout={config['dropout']},
                                      bidirectional={config['bidirectional']},
                                      use_gru={config['use_gru']},
                                      fc_num_layers={config['fc_num_layers']},
                                      fc_hidden_dims={config['fc_hidden_dims']},
                                      fc_activation='{config['fc_activation']}',
                                      fc_dropout={config['fc_dropout']},
                                      use_batch_norm={config['use_batch_norm']}).to(self.device)
        checkpoint = torch.load('{model_filename}', map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.current_seq_ix = None
        self.sequence_history = []
    def predict(self, data_point: DataPoint) -> np.ndarray:
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

    solution_path = submission_dir / 'solution.py'
    with open(solution_path, 'w') as f:
        f.write(solution_code)
    print(f"  Generated: solution.py")

    # Copy utils.py
    utils_src = Path('../../wunderfund/competition_package/utils.py')
    utils_dst = submission_dir / 'utils.py'
    shutil.copy(utils_src, utils_dst)
    print(f"  Copied: utils.py")

    # Save metadata
    metadata = {
        'config_name': config_name,
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
    print(f"  Saved: metadata.json")

    print("=" * 70)
    print(f"\n✅ Submission created: {submission_dir}")

    return submission_dir


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM with attention configuration and create submission'
    )
    parser.add_argument(
        '--config',
        type=str,
        choices=list(CONFIGS.keys()),
        default='config1',
        help='Predefined configuration to use'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=40,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Use smaller test dataset'
    )

    args = parser.parse_args()

    # Setup device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("\n" + "=" * 70)
    print("LSTM WITH ATTENTION TRAINING AND SUBMISSION CREATION")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    print(f"Max epochs: {args.max_epochs}")
    print("=" * 70)

    config = CONFIGS[args.config].copy()

    print("\nLoading data...")
    if args.test:
        train_df = pd.read_parquet('data/tiny_train.parquet')
        val_df = pd.read_parquet('data/tiny_val.parquet')
    else:
        full_df = pd.read_parquet('../competition_package/datasets/train.parquet')
        all_seqs = full_df['seq_ix'].unique()
        np.random.seed(42)
        train_seqs = np.random.choice(all_seqs, size=int(0.8 * len(all_seqs)), replace=False)
        val_seqs = np.array([s for s in all_seqs if s not in train_seqs])
        train_df = full_df[full_df['seq_ix'].isin(train_seqs)].reset_index(drop=True)
        val_df = full_df[full_df['seq_ix'].isin(val_seqs)].reset_index(drop=True)

    feature_cols = [col for col in train_df.columns if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]

    print(f"Train: {len(train_df):,} rows, {train_df['seq_ix'].nunique()} sequences")
    print(f"Val:   {len(val_df):,} rows, {val_df['seq_ix'].nunique()} sequences")
    print(f"Features: {len(feature_cols)}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_time = time.time()

    result = train_model(config, train_df, val_df, feature_cols, device, max_epochs=args.max_epochs)

    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time/60:.1f} minutes")

    submission_dir = create_submission(config, result, feature_cols, args.config, timestamp)

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    print(f"\nSubmission ready at: {submission_dir}")
    print(f"\nTo create a zip file:")
    print(f"  cd ../submissions")
    print(f"  zip -r {submission_dir.name}.zip {submission_dir.name}/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
