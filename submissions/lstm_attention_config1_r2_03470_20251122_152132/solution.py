"""
GRU Solution with Self-Attention - config1
Timestamp: 20251122_152132

Best validation R²: 0.346970
Training R²: 0.410966
Overfitting gap: 0.063996

Architecture:
- GRU (Bidirectional) with Self-Attention
- 3 layers, 128 hidden units
- Lookback: 50 timesteps
- Dropout: 0.3
- FC layers: 2, activation: relu
- Parameters: 857,889

Training:
- Optimizer: ADAM
- Learning rate: 0.0001
- Weight decay: 0.0
- Batch size: 256
- Best epoch: 31/36
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
                 hidden_size=128,
                 num_layers=3,
                 dropout=0.3,
                 bidirectional=True,
                 use_gru=True,
                 fc_num_layers=2,
                 fc_hidden_dims=256,
                 fc_activation='relu',
                 fc_dropout=0.0,
                 use_batch_norm=True):
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
        attention_out = self.attention(rnn_out)
        return self.output_head(attention_out)

class PredictionModel:
    def __init__(self, lookback=50, n_features=32):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')
        self.model = LSTMWithAttention(input_size=n_features,
                                      hidden_size=128,
                                      num_layers=3,
                                      dropout=0.3,
                                      bidirectional=True,
                                      use_gru=True,
                                      fc_num_layers=2,
                                      fc_hidden_dims=256,
                                      fc_activation='relu',
                                      fc_dropout=0.0,
                                      use_batch_norm=True).to(self.device)
        checkpoint = torch.load('model_best.pt', map_location=self.device, weights_only=False)
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
