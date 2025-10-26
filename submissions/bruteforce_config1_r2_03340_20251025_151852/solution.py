"""
LSTM Solution - config1
Timestamp: 20251025_151852

Best validation R²: 0.333983
Training R²: 0.390997
Overfitting gap: 0.057013

Architecture:
- LSTM (Unidirectional)
- 1 layers, 32 hidden units
- Lookback: 50 timesteps
- Dropout: 0.3
- FC layers: 1, activation: relu
- Parameters: 9,504

Training:
- Optimizer: ADAM
- Learning rate: 0.0001
- Weight decay: 0.0
- Batch size: 32
- Best epoch: 39/40
"""

import os
import numpy as np
import torch
import torch.nn as nn
from utils import DataPoint


class FlexibleRNN(nn.Module):
    """Flexible RNN-based sequence predictor."""

    def __init__(
        self,
        input_size,
        hidden_size=32,
        num_layers=1,
        dropout=0.3,
        bidirectional=False,
        use_gru=False,
        fc_num_layers=1,
        fc_hidden_dims=256,
        fc_activation='relu',
        fc_dropout=0.0,
        use_batch_norm=True
    ):
        super(FlexibleRNN, self).__init__()

        # Choose RNN type
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        self.output_head = self._build_output_head(
            rnn_output_size, input_size, fc_num_layers,
            fc_hidden_dims, fc_activation, fc_dropout, use_batch_norm
        )

    def _build_output_head(
        self, input_dim, output_dim, num_layers,
        hidden_dims, activation, dropout, use_batch_norm
    ):
        """Build fully connected output layers."""
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
    Wrapper for LSTM model matching competition API.

    Maintains lookback window and predicts next state.
    Falls back to simple average when insufficient history.
    """

    def __init__(self, lookback=50, n_features=32):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')

        # Initialize model
        self.model = FlexibleRNN(input_size=n_features).to(self.device)

        # Load trained weights
        checkpoint = torch.load('model_best.pt', map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Sequence memory
        self.current_seq_ix = None
        self.sequence_history = []

    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Generate prediction for next timestep.

        Args:
            data_point: Current observation

        Returns:
            np.ndarray: Predicted next state (if required)
        """
        # Reset on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Add observation
        self.sequence_history.append(data_point.state.copy())

        if not data_point.need_prediction:
            return None

        # Fallback when insufficient context
        if len(self.sequence_history) < self.lookback:
            return np.mean(self.sequence_history, axis=0)

        # Prepare input sequence
        sequence = np.array(self.sequence_history[-self.lookback:], dtype=np.float32)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
            prediction = prediction.cpu().numpy()[0]

        return prediction


# ================================================================
# Optional: Local evaluation using ScorerStepByStep
# ================================================================
if __name__ == "__main__":
    try:
        from utils import ScorerStepByStep
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(CURRENT_DIR, "datasets", "train.parquet")

        print("Running local evaluation (this may take a while)...")
        model = PredictionModel()
        scorer = ScorerStepByStep("datasets/train.parquet")
        results = scorer.score(model)

        print("\nResults:")
        print(f"Mean R² across all features: {results['mean_r2']:.6f}")
        print("\nR² for first 5 features:")
        for i in range(min(5, len(scorer.features))):
            print(f"  {i}: {results[scorer.features[i]]:.6f}")

    except Exception as e:
        print("Local evaluation skipped or failed:", e)
        print("You can still use this solution.py for submission as long as models_hybrid/ and artifacts/ are present.")
