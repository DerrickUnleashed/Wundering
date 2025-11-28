"""
GRU Solution - Brute Force Search
Config ID: 1
Timestamp: 20251127_234159

Best validation R²: 0.378178
Training R²: 0.400634
Overfitting gap: 0.022456

Architecture:
- GRU (Bidirectional)
- 1 layers, 256 hidden units
- Lookback: 100 timesteps
- Dropout: 0.1
- FC layers: 2, activation: relu
- Parameters: 585,504

Training:
- Optimizer: ADAM
- Learning rate: 0.0001
- Weight decay: 0.0001
- Batch size: 256
- Best epoch: 38/48
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from utils import DataPoint
import warnings
warnings.filterwarnings("ignore")


class FlexibleRNN(nn.Module):
    """Flexible RNN-based sequence predictor."""

    def __init__(
        self,
        input_size,
        hidden_size=256,
        num_layers=1,
        dropout=0.1,
        bidirectional=True,
        use_gru=True,
        fc_num_layers=2,
        fc_hidden_dims=256,
        fc_activation="relu",
        fc_dropout=0.2,
        use_batch_norm=True
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
        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        return self.output_head(last_output)


class PredictionModel:
    """
    Wrapper for GRU model matching competition API.

    Maintains lookback window and predicts next state.
    Falls back to simple average when insufficient history.
    """

    def __init__(self, lookback=100, n_features=32):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')  # Use CPU for submission

        # Initialize model
        self.model = FlexibleRNN(input_size=n_features).to(self.device)

        # Load trained weights
        checkpoint = torch.load('model_best.pt', map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load preprocessing
        with open('preprocessing.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        self.scaler = preprocessing['scaler']
        self.lower = preprocessing['lower']
        self.upper = preprocessing['upper']

        # Sequence state
        self.current_seq_ix = None
        self.sequence_history = []

    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Safe, NaN-proof prediction function.
        Follows training-time preprocessing exactly:
        1. clip
        2. scale
        """

        # Reset if new sequence begins
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Append current raw state
        x = np.asarray(data_point.state, dtype=np.float32)

        # Sanity check: replace inf/nan in the input
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        self.sequence_history.append(x)

        # If prediction not needed → return None
        if not data_point.need_prediction:
            return None

        # Not enough history → fallback to average
        if len(self.sequence_history) < self.lookback:
            return np.mean(self.sequence_history, axis=0)

        # ----------------------------------------------------------------------
        #  Build sequence from last LOOKBACK steps (raw)
        # ----------------------------------------------------------------------
        seq = np.asarray(self.sequence_history[-self.lookback:], dtype=np.float32)

        # ----------------------------------------------------------------------
        # 1) CLIP  (vectorized, no pandas)
        #    seq_clipped[t, f] = np.clip(seq[t, f], lower[f], upper[f])
        # ----------------------------------------------------------------------
        seq = np.clip(seq, self.lower, self.upper)

        # ----------------------------------------------------------------------
        # 2) SCALE (same StandardScaler used during training)
        # ----------------------------------------------------------------------
        # StandardScaler expects shape (N, features)
        seq = self.scaler.transform(seq)

        # Final NaN check before model
        if not np.isfinite(seq).all():
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        # Format for model
        tensor = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)

        # ----------------------------------------------------------------------
        # 3) Model forward
        # ----------------------------------------------------------------------
        with torch.no_grad():
            pred = self.model(tensor)
            pred = pred.cpu().numpy()[0]

        # Final safety clean
        if not np.isfinite(pred).all():
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        return pred


print("=== Loading model ===")
model = PredictionModel()

print("=== Generating synthetic sequence ===")

# 200 steps of random data matching real feature stats
# This is ONLY for testing — does not affect competition
sequence_len = 200
n_features = model.n_features

# uniform range approx original range (-5,5)
fake_data = np.random.uniform(-5, 5, (sequence_len, n_features))

print("=== Running through model ===")
outputs = []

for i in range(sequence_len):
    dp = DataPoint(
        seq_ix=0,
        step_in_seq=i,
        state=fake_data[i],
        need_prediction=(i >= 1)   # predict starting at step 1
    )

    pred = model.predict(dp)

    if pred is not None:
        outputs.append(pred)

        # NA sanity check
        if np.isnan(pred).any():
            print(f"❌ NaN detected at step {i}")
            print(pred)
            break

print("=== Test finished ===")
if outputs and not np.isnan(outputs).any():
    print("✅ SUCCESS: No NaNs across all predictions.")
else:
    print("⚠️ No predictions or NaN found.")

