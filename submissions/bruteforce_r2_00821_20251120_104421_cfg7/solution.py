"""
LSTM Encoder-Decoder Solution - Brute Force Search
Config ID: 7
Timestamp: 20251120_104421

Best validation R²: 0.082055
Training R²: 0.371347
Overfitting gap: 0.289292

Architecture (factory):
- Encoder hidden: 128
- Decoder hidden: 64
- Encoder layers: 1
- Decoder layers: 1
- Bidirectional encoder: True
- DropConnect p: 0.1
- Attention heads: 4, attn_dim: None

Lookback: 50
Parameters: 599,456

Training:
- Optimizer: ADAM
- Learning rate: 0.0005
- Weight decay: 0.0001
- Batch size: 128
- Best epoch: 20/27
"""

import numpy as np
import torch
import torch.nn as nn
from utils import DataPoint

# NOTE: This submission loads the saved checkpoint and expects
# the competition runtime to supply DataPoint objects as used before.

class PredictionModel:
    def __init__(self, lookback=50, n_features=32):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')

        # We instantiate a minimal placeholder model whose architecture
        # mirrors training-time factory. For the submission the actual
        # model weights are loaded from model_best.pt saved alongside.
        # For deterministic simplicity the submission uses a small wrapper.

        # Lazy load checkpoint then build a model architecture if needed.
        ckpt = torch.load('model_best.pt', map_location=self.device)
        # We do not attempt to replicate the full factory here; instead,
        # for inference we load the saved state dict into a simple compatible container.
        # A minimal model definition consistent with the saved state is required here.
        # For brevity, the submission uses the same design as during training:
        try:
            # If the saved checkpoint contains a full 'config' we can reconstruct the model
            cfg = ckpt.get('config', {})
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
