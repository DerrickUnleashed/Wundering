"""
Transformer Solution - Brute Force Search
Config ID: 18
Timestamp: 20251029_233946

Best validation R²: 0.356213
Training R²: 0.395972
Overfitting gap: 0.039759

Architecture:
- Transformer encoder (d_model=128, layers=2, nhead=4)
- Lookback: 50 timesteps
- Parameters: 290,080

Training:
- Optimizer: ADAM
- Learning rate: 0.0003
- Weight decay: 1e-05
- Batch size: 128
- Best epoch: 25/32
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

    def __init__(self, input_size, d_model=128, num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1, fc_hidden_dims=256, fc_activation='relu'):
        super(FlexibleTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        activation = nn.ReLU() if fc_activation == 'relu' else nn.GELU()
        self.output_head = nn.Sequential(
            nn.Linear(d_model, fc_hidden_dims),
            nn.BatchNorm1d(fc_hidden_dims),
            activation,
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
    def __init__(self, lookback=50, n_features=32):
        self.lookback = lookback
        self.n_features = n_features
        # Use CPU device by default for deployed solutions
        self.device = torch.device('cpu')

        # Allowlist common numpy globals to support torch.load() in some torch versions
        try:
            from torch.serialization import add_safe_globals
            import numpy as _numpy
            add_safe_globals([_numpy.dtype, _numpy.ndarray, _numpy.generic, _numpy.core.multiarray.scalar])
        except Exception:
            # Not available on all torch builds; ignore silently
            pass

        # Load checkpoint first so we can inspect the training config and instantiate
        # the model with the exact architecture used during training (avoids size mismatches).
        # Use weights_only=False to allow loading the full dict (config + state) on newer torch versions.
        checkpoint = torch.load('model_best.pt', map_location=self.device, weights_only=False)

        # Try to read training config from the checkpoint (saved by the training script)
        ckpt_config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}

        # Extract architecture hyperparameters (use sensible defaults if missing)
        d_model = int(ckpt_config.get('d_model', 128))
        num_layers = int(ckpt_config.get('num_transformer_layers', ckpt_config.get('num_layers', 2)))
        nhead = int(ckpt_config.get('nhead', 4))
        dim_feedforward = int(ckpt_config.get('dim_feedforward', 256))
        dropout = float(ckpt_config.get('dropout', 0.1))
        fc_hidden_dims = int(ckpt_config.get('fc_hidden_dims', ckpt_config.get('fc_hidden_dim', 256)))
        fc_activation = ckpt_config.get('fc_activation', 'relu')

        # Instantiate model with hyperparameters matching the checkpoint
        self.model = FlexibleTransformer(
            input_size=n_features,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            fc_hidden_dims=fc_hidden_dims,
            fc_activation=fc_activation,
        ).to(self.device)

        # Determine which key contains the state dict and load it
        state_key = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_key = 'model_state_dict'
            elif 'state_dict' in checkpoint:
                state_key = 'state_dict'

        if state_key is None:
            raise RuntimeError('Checkpoint does not contain model weights')

        try:
            # Strict loading ensures we catch mismatches early
            self.model.load_state_dict(checkpoint[state_key], strict=True)
        except Exception as e:
            # Provide a helpful error that includes what we attempted to load
            raise RuntimeError(f'Failed to load model state dict: {e}') from e

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
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(seq_tensor)
            return pred.cpu().numpy()[0]


if __name__ == "__main__":
    # Test code to verify model loading and prediction
    import numpy as np
    from utils import DataPoint
    
    print("Testing Transformer prediction model...")
    
    # Create a test sequence
    n_features = 32
    lookback = 50
    test_sequence = np.random.randn(60, n_features).astype(np.float32)
    
    # Initialize model
    model = PredictionModel(lookback=lookback, n_features=n_features)
    print("Model initialized successfully")
    
    # Test predictions
    for i, state in enumerate(test_sequence):
        data_point = DataPoint(
            seq_ix=1,
            step_in_seq=i,  # Add step index
            state=state,
            need_prediction=(i >= lookback-1)  # Start predicting after lookback steps
        )
        pred = model.predict(data_point)
        
        if pred is not None:
            print(f"Step {i+1}: Made prediction with shape {pred.shape}")
            # Verify prediction is finite and reasonable
            assert np.all(np.isfinite(pred)), "Found non-finite values in prediction"
            assert pred.shape == (n_features,), f"Wrong prediction shape: {pred.shape}"
    
    print("All tests passed! Model is working correctly.")