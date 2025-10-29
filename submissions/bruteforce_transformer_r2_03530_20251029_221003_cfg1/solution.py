"""
Transformer Solution - Brute Force Search
Config ID: 1
Timestamp: 20251029_221003

Best validation R²: 0.352990
Training R²: 0.391162
Overfitting gap: 0.038172

Architecture:
- Transformer encoder (d_model=128, layers=2, nhead=4)
- Lookback: 50 timesteps
- Parameters: 310,944

Training:
- Optimizer: ADAM
- Learning rate: 0.0001
- Weight decay: 0.0
- Batch size: 256
- Best epoch: 27/30
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
    def __init__(self, input_size, d_model=128, num_layers=2, nhead=8, dim_feedforward=256, dropout=0.1, fc_hidden_dims=256, fc_activation='relu'):
        super(FlexibleTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Match training output head: Linear -> BatchNorm1d -> Activation -> Dropout -> Linear
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
        # No need to permute with batch_first=True
        out = self.transformer(x)
        last = out[:,-1,:]  # Get last sequence element for each batch
        return self.output_head(last)


class PredictionModel:
    def __init__(self, lookback=50, n_features=32):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')
        self.model = FlexibleTransformer(input_size=n_features, d_model=128, num_layers=2, nhead=4)
        # Allow numpy globals for model loading
        from torch.serialization import add_safe_globals
        import numpy.core.multiarray
        import numpy
        import numpy.dtypes
        
        # Add all required numpy types to safe globals
        safe_types = [
            numpy.core.multiarray.scalar,
            numpy.dtype,
            numpy.ndarray,
            numpy.generic,
            numpy.dtypes.Float64DType
        ]
        add_safe_globals(safe_types)
        
        try:
            # First try loading with weights_only=False since we trust our own checkpoint
            checkpoint = torch.load('model_best.pt', map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            # If the model_state_dict key doesn't exist, try state_dict
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError("Could not find model weights in checkpoint")
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
