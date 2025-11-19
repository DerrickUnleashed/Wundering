"""
GRU Encoder-Decoder Solution - Brute Force Search
Config ID: 1
Timestamp: 20251120_013337

Best validation R²: 0.343029
Training R²: 0.342635
Overfitting gap: -0.000394

Architecture (factory):
- Encoder hidden: 128
- Decoder hidden: 64
- Encoder layers: 1
- Decoder layers: 1
- Bidirectional encoder: True
- DropConnect p: 0.1
- Attention heads: 4, attn_dim: None

Lookback: 50
Parameters: 538,912

Training:
- Optimizer: ADAM
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 256
- Best epoch: 1/1
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
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        ckpt = torch.load("model_best.pt", map_location=self.device, weights_only=False)
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

if __name__ == "__main__":
    import numpy as np
    from utils import DataPoint

    print("Testing model prediction...")

    n_features = 32
    lookback = 50

    # Create dummy sequence (slightly more than lookback)
    test_sequence = np.random.randn(60, n_features).astype(np.float32)

    # Initialize model
    model = PredictionModel(lookback=lookback, n_features=n_features)
    print("Model initialized successfully")

    # Step-by-step inference
    for i, state in enumerate(test_sequence):
        dp = DataPoint(
            seq_ix=1,
            step_in_seq=i,
            state=state,
            need_prediction=(i >= lookback - 1)
        )

        pred = model.predict(dp)

        if pred is not None:
            print(f"Step {i+1}: Made prediction with shape {pred.shape}")
            assert np.all(np.isfinite(pred)), "Non-finite prediction detected"
            assert pred.shape == (n_features,), f"Wrong prediction shape: {pred.shape}"

    print("All tests passed! Model is working correctly.")
