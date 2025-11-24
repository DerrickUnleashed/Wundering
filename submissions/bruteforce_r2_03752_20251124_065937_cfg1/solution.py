"""
GRU Solution - Brute Force Search
Config ID: 1
Timestamp: 20251124_065937

Best validation R²: 0.375249
Training R²: 0.398764
Overfitting gap: 0.023515

Architecture:
- GRU (Bidirectional)
- 1 layers, 128 hidden units
- Lookback: 100 timesteps
- Dropout: 0.1
- FC layers: 2, activation: relu
- Parameters: 198,944

Training:
- Optimizer: ADAM
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 256
- Best epoch: 43/58
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
        hidden_size=128,
        num_layers=1,
        dropout=0.1,
        bidirectional=True,
        use_gru=True,
        fc_num_layers=2,
        fc_hidden_dims=256,
        fc_activation='relu',
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
