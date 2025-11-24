"""
Encoder-Decoder GRU Solution - Brute Force Search
Config ID: 1
Timestamp: 20251123_222612

Best validation R²: 0.373315
Training R²: 0.384691
Overfitting gap: 0.011376

Architecture:
- Encoder-Decoder GRU (Bidirectional)
- Encoder: 1 layers, 128 hidden
- Decoder: 1 layers, 64 hidden
- Lookback: 100 timesteps
- Dropout: 0.2, DropConnect: 0.1
- Attention dim: 64
- Parameters: 497,952

Training:
- Optimizer: ADAMW
- Learning rate: 0.0005
- Weight decay: 0.0
- Batch size: 256
- Best epoch: 10/20
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils import DataPoint


class DropConnectLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, p: float = 0.1):
        super().__init__(in_features, out_features, bias=bias)
        self.p = float(p)

    def forward(self, input):
        if self.training and self.p > 0.0:
            mask = torch.bernoulli((1.0 - self.p) * torch.ones_like(self.weight)).to(self.weight.device)
            w = self.weight * mask
        else:
            w = self.weight * (1.0 - self.p)
        return F.linear(input, w, self.bias)


class AttentionBlock(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.Wq = nn.Linear(dec_dim, attn_dim, bias=False)
        self.Wk = nn.Linear(enc_dim, attn_dim, bias=False)
        self.Wv = nn.Linear(enc_dim, attn_dim, bias=False)
        self.out = nn.Linear(attn_dim, dec_dim)

    def forward(self, enc_seq, dec_vec):
        Q = self.Wq(dec_vec).unsqueeze(1)
        K = self.Wk(enc_seq)
        V = self.Wv(enc_seq)
        scores = torch.softmax((Q * K).sum(-1), dim=-1).unsqueeze(-1)
        context = (scores * V).sum(1)
        out = self.out(context)
        return out


class EncoderDecoderLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        encoder_hidden: int = 128,
        decoder_hidden: int = 64,
        encoder_num_layers: int = 1,
        decoder_num_layers: int = 1,
        bidirectional_encoder: bool = True,
        dropout: float = 0.2,
        dropconnect_p: float = 0.1,
        attn_dim: Optional[int] = 64,
        use_gru: bool = True,
        use_batch_norm: bool = True,
        **kwargs, # absorb extra keys
    ):
        super().__init__()
        self.input_size = input_size
        self.encoder_hidden = int(encoder_hidden)
        self.decoder_hidden = int(decoder_hidden)
        self.bidirectional_encoder = bool(bidirectional_encoder)
        self.encoder_num_layers = int(encoder_num_layers)
        self.decoder_num_layers = int(decoder_num_layers)
        self.dropout = float(dropout)
        self.dropconnect_p = float(dropconnect_p)
        self.use_gru = bool(use_gru)
        self.use_batch_norm = bool(use_batch_norm)

        rnn_class = nn.GRU if self.use_gru else nn.LSTM
        self.encoder = rnn_class(
            input_size, self.encoder_hidden, num_layers=self.encoder_num_layers,
            batch_first=True, dropout=self.dropout if self.encoder_num_layers > 1 else 0.0,
            bidirectional=self.bidirectional_encoder
        )

        self.encoder_dim = self.encoder_hidden * (2 if self.bidirectional_encoder else 1)
        mid_dim = max(self.encoder_dim, 512)
        self.enc_fc1 = DropConnectLinear(self.encoder_dim, mid_dim, p=self.dropconnect_p)
        self.enc_bn1 = nn.BatchNorm1d(mid_dim) if self.use_batch_norm else nn.Identity()
        self.enc_fc2 = DropConnectLinear(mid_dim, self.encoder_dim, p=self.dropconnect_p)
        self.enc_bn2 = nn.BatchNorm1d(self.encoder_dim) if self.use_batch_norm else nn.Identity()
        self.enc_act = nn.ReLU()
        self.enc_dropout = nn.Dropout(p=self.dropout)

        effective_attn_dim = attn_dim if attn_dim is not None else min(128, max(64, self.encoder_dim // 2))
        self.attention = AttentionBlock(self.encoder_dim, self.decoder_hidden, effective_attn_dim)

        self.decoder = rnn_class(
            self.encoder_dim, self.decoder_hidden, num_layers=self.decoder_num_layers,
            batch_first=True, dropout=self.dropout if self.decoder_num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.output_head = nn.Sequential(
            nn.Linear(self.decoder_hidden, max(self.decoder_hidden, self.input_size)),
            nn.ReLU(), nn.Dropout(p=self.dropout),
            nn.Linear(max(self.decoder_hidden, self.input_size), self.input_size)
        )

    def forward(self, x):
        enc_seq, _ = self.encoder(x)
        last_enc = enc_seq[:, -1, :]
        z = self.enc_fc1(last_enc)
        z = self.enc_bn1(z)
        z = self.enc_act(z)
        z = self.enc_dropout(z)
        z2 = self.enc_fc2(z)
        z2 = self.enc_bn2(z2)
        z2 = self.enc_act(z2)
        z2 = self.enc_dropout(z2)
        enc_final = last_enc + z2
        seq_len = x.size(1)
        dec_in = enc_final.unsqueeze(1).repeat(1, seq_len, 1)
        dec_seq, _ = self.decoder(dec_in)
        last_dec = dec_seq[:, -1, :]
        attn_ctx = self.attention(enc_seq, last_dec)
        dec_fused = last_dec + attn_ctx
        out = self.output_head(dec_fused)
        return out


class PredictionModel:
    """
    Wrapper for Encoder-Decoder GRU model matching competition API.
    """

    def __init__(self, lookback=100, n_features=32):
        self.lookback = lookback
        self.n_features = n_features
        self.device = torch.device('cpu')

        # Initialize model with all config params
        self.model = EncoderDecoderLSTM(input_size=n_features).to(self.device)

        # Load trained weights
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

        sequence = np.array(
            self.sequence_history[-self.lookback:],
            dtype=np.float32
        )
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(sequence_tensor)
            prediction = prediction.cpu().numpy()[0]

        return prediction
