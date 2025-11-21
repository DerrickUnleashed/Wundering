"""
Enhanced LSTM/GRU Models with Flexible Architecture

Supports:
- LSTM or GRU
- Bidirectional option
- Multi-layer output head with activations, dropout, batch norm
- Flexible regularization
- GRU/LSTM + Multi-Head Attention (hybrid)
- Transformer architecture (PatchTST-style)
"""

import torch
import torch.nn as nn
import math


class FlexibleRNN(nn.Module):
    """Flexible RNN-based sequence predictor with extensive configuration options."""

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
        fc_dropout=0.2,
        use_batch_norm=False
    ):
        """
        Initialize flexible RNN model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension for RNN
            num_layers: Number of RNN layers
            dropout: Dropout for RNN (applied between layers if num_layers > 1)
            bidirectional: Whether to use bidirectional RNN
            use_gru: If True, use GRU; otherwise LSTM
            fc_num_layers: Number of fully connected layers in output head
            fc_hidden_dims: Hidden dimensions for FC layers (if fc_num_layers > 1)
            fc_activation: Activation function ('relu', 'tanh', 'gelu', 'leaky_relu')
            fc_dropout: Dropout rate for FC layers
            use_batch_norm: Whether to use batch normalization in FC layers
        """
        super(FlexibleRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_gru = use_gru

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

        # Calculate RNN output size (doubled if bidirectional)
        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        # Build output head (FC layers)
        self.output_head = self._build_output_head(
            input_dim=rnn_output_size,
            output_dim=input_size,
            num_layers=fc_num_layers,
            hidden_dims=fc_hidden_dims,
            activation=fc_activation,
            dropout=fc_dropout,
            use_batch_norm=use_batch_norm
        )

    def _build_output_head(
        self, input_dim, output_dim, num_layers,
        hidden_dims, activation, dropout, use_batch_norm
    ):
        """Build fully connected output head."""
        layers = []

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2)
        }
        act_fn = activation_map.get(activation, nn.ReLU())

        if num_layers == 1:
            # Simple single linear layer
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Multi-layer head
            current_dim = input_dim

            # Hidden layers
            for i in range(num_layers - 1):
                layers.append(nn.Linear(current_dim, hidden_dims))

                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dims))

                layers.append(act_fn)

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

                current_dim = hidden_dims

            # Final output layer
            layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, input_size)
        """
        # RNN forward pass
        rnn_out, _ = self.rnn(x)

        # Take last timestep output
        last_output = rnn_out[:, -1, :]

        # Pass through output head
        output = self.output_head(last_output)

        return output

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        """Return model configuration as dictionary."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'use_gru': self.use_gru,
            'total_parameters': self.count_parameters()
        }


class RNNWithAttention(nn.Module):
    """
    RNN (GRU/LSTM) + Multi-Head Attention hybrid architecture.

    Combines the temporal modeling of RNNs with the feature interaction
    capabilities of multi-head attention.
    """

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
        use_gru=True,
        # Attention parameters
        use_attention=True,
        attention_heads=4,
        attention_dropout=0.1,
        # Output head parameters
        fc_num_layers=2,
        fc_hidden_dims=256,
        fc_activation='relu',
        fc_dropout=0.2,
        use_batch_norm=True
    ):
        """
        Initialize RNN + Attention hybrid model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension for RNN
            num_layers: Number of RNN layers
            dropout: Dropout for RNN
            bidirectional: Whether to use bidirectional RNN
            use_gru: If True, use GRU; otherwise LSTM
            use_attention: Whether to add attention layer
            attention_heads: Number of attention heads
            attention_dropout: Dropout for attention
            fc_num_layers: Number of FC layers in output head
            fc_hidden_dims: Hidden dimensions for FC layers
            fc_activation: Activation function
            fc_dropout: Dropout rate for FC layers
            use_batch_norm: Whether to use batch normalization
        """
        super(RNNWithAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_gru = use_gru
        self.use_attention = use_attention

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

        # Calculate RNN output size
        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        # Multi-head attention layer (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=rnn_output_size,
                num_heads=attention_heads,
                dropout=attention_dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(rnn_output_size)
            self.attention_dropout = nn.Dropout(attention_dropout)

        # Build output head
        self.output_head = self._build_output_head(
            input_dim=rnn_output_size,
            output_dim=input_size,
            num_layers=fc_num_layers,
            hidden_dims=fc_hidden_dims,
            activation=fc_activation,
            dropout=fc_dropout,
            use_batch_norm=use_batch_norm
        )

    def _build_output_head(
        self, input_dim, output_dim, num_layers,
        hidden_dims, activation, dropout, use_batch_norm
    ):
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
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, input_size)
        """
        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_size * directions)

        # Apply attention if enabled
        if self.use_attention:
            # Self-attention over time dimension
            attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
            # Residual connection + layer norm
            rnn_out = self.attention_norm(rnn_out + self.attention_dropout(attn_out))

        # Take last timestep output
        last_output = rnn_out[:, -1, :]

        # Pass through output head
        output = self.output_head(last_output)

        return output

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        """Return model configuration as dictionary."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'use_gru': self.use_gru,
            'use_attention': self.use_attention,
            'total_parameters': self.count_parameters()
        }


def create_model_from_config(config, input_size):
    """
    Create model from hyperparameter configuration.

    Args:
        config: Dictionary with hyperparameters
        input_size: Number of input features

    Returns:
        Initialized model
    """
    # Check if attention model is requested
    if config.get('use_attention', False):
        return RNNWithAttention(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional'],
            use_gru=config['use_gru'],
            use_attention=True,
            attention_heads=config.get('attention_heads', 4),
            attention_dropout=config.get('attention_dropout', 0.1),
            fc_num_layers=config['fc_num_layers'],
            fc_hidden_dims=config.get('fc_hidden_dims', 128),
            fc_activation=config['fc_activation'],
            fc_dropout=config['fc_dropout'],
            use_batch_norm=config['use_batch_norm']
        )
    else:
        # Standard FlexibleRNN
        return FlexibleRNN(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional'],
            use_gru=config['use_gru'],
            fc_num_layers=config['fc_num_layers'],
            fc_hidden_dims=config.get('fc_hidden_dims', 128),
            fc_activation=config['fc_activation'],
            fc_dropout=config['fc_dropout'],
            use_batch_norm=config['use_batch_norm']
        )

# config = {
#         # Config 1 from bruteforce search - achieved 0.3414 val R²
#         'lookback': 50,
#         'hidden_size': 128,
#         'num_layers': 3,
#         'dropout': 0.3,
#         'bidirectional': False,
#         'use_gru': True,
#         'fc_num_layers': 2,
#         'fc_hidden_dims': 256,
#         'fc_activation': 'relu',
#         'fc_dropout': 0.0,
#         'use_batch_norm': True,
#         'batch_size': 256,
#         'lr': 0.0001,
#         'weight_decay': 0.0,
#         'grad_clip': 0.5,
#         'optimizer': 'adam',
#         'lr_patience': 3,
#     }

# models.py
"""
Enhanced Encoder-Decoder LSTM model with:
- Bidirectional encoder
- DropConnect in FC layers
- Additive attention between encoder outputs and decoder final state
- Residual connections
- Backwards-compatible factory: create_model_from_config(config, input_size)

This file is intended to be a drop-in replacement for the original models module
used by the stored brute-force hyperparameter search script. The factory function
`create_model_from_config(config, input_size)` preserves the same signature so
the training script requires no changes; only the architecture is updated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DropConnectLinear(nn.Linear):
    """
    DropConnect applied to the weight matrix of a linear layer.
    During training: random binary mask applied to weights each forward pass.
    During eval: weights scaled by (1 - p).
    """
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
    """
    Simple attention that computes attention over encoder time-steps
    using decoder query (vector). Produces a context vector of size attn_dim,
    then projected back to dec_dim.
    """
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.Wq = nn.Linear(dec_dim, attn_dim, bias=False)
        self.Wk = nn.Linear(enc_dim, attn_dim, bias=False)
        self.Wv = nn.Linear(enc_dim, attn_dim, bias=False)
        self.out = nn.Linear(attn_dim, dec_dim)

    def forward(self, enc_seq, dec_vec):
        # enc_seq: (B, T, enc_dim)
        # dec_vec: (B, dec_dim)
        Q = self.Wq(dec_vec).unsqueeze(1)    # (B,1,attn)
        K = self.Wk(enc_seq)                 # (B,T,attn)
        V = self.Wv(enc_seq)                 # (B,T,attn)

        # Dot-product attention (scaled not necessary for learned projections here)
        scores = torch.softmax((Q * K).sum(-1), dim=-1).unsqueeze(-1)  # (B,T,1)
        context = (scores * V).sum(1)                                 # (B,attn)
        out = self.out(context)                                       # (B,dec_dim)
        return out


class EncoderDecoderLSTM(nn.Module):
    """
    Encoder-Decoder LSTM (seq -> vector -> seq) with:
    - Bidirectional encoder (optional)
    - Encoder FC block with DropConnect and residuals
    - Decoder LSTM consuming repeated encoder vector
    - Attention between encoder sequence outputs and decoder final state
    - Minimal output head mapping decoder hidden -> input feature dim
    """

    def __init__(
        self,
        input_size: int,
        encoder_hidden: int = 256,
        decoder_hidden: int = 64,
        encoder_num_layers: int = 1,
        decoder_num_layers: int = 1,
        bidirectional_encoder: bool = True,
        dropout: float = 0.2,
        dropconnect_p: float = 0.1,
        attn_heads: int = 4,
        attn_dim: Optional[int] = None,
        use_gru: bool = False,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_size: number of input features (e.g., 32)
            encoder_hidden: hidden units in encoder LSTM per direction
            decoder_hidden: hidden units in decoder LSTM
            encoder_num_layers: number of stacked encoder layers
            decoder_num_layers: number of stacked decoder layers
            bidirectional_encoder: whether encoder is bidirectional
            dropout: dropout inside RNN modules (applied when num_layers>1)
            dropconnect_p: dropconnect probability for FC layers (0.0 - 0.5 typical)
            attn_heads: not used as multihead; kept for config compatibility (attn_dim controls)
            attn_dim: dimensionality of attention internal projection (defaults to min(128, encoder_dim//2))
            use_gru: if True uses GRU for encoder/decoder; else LSTM
            use_batch_norm: whether to use BatchNorm in encoder fc block
        """
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

        # Encoder RNN
        rnn_class = nn.GRU if self.use_gru else nn.LSTM
        self.encoder = rnn_class(
            input_size,
            self.encoder_hidden,
            num_layers=self.encoder_num_layers,
            batch_first=True,
            dropout=self.dropout if self.encoder_num_layers > 1 else 0.0,
            bidirectional=self.bidirectional_encoder
        )

        # encoder output dim (hidden per direction * n_dirs)
        self.encoder_dim = self.encoder_hidden * (2 if self.bidirectional_encoder else 1)

        # Encoder FC block (applied to last timestep encoder vector)
        # We'll project encoder_dim -> mid_dim -> encoder_dim for residual
        mid_dim = max(self.encoder_dim, 512)
        self.enc_fc1 = DropConnectLinear(self.encoder_dim, mid_dim, p=self.dropconnect_p)
        self.enc_bn1 = nn.BatchNorm1d(mid_dim) if self.use_batch_norm else nn.Identity()
        self.enc_fc2 = DropConnectLinear(mid_dim, self.encoder_dim, p=self.dropconnect_p)
        self.enc_bn2 = nn.BatchNorm1d(self.encoder_dim) if self.use_batch_norm else nn.Identity()
        self.enc_act = nn.ReLU()
        self.enc_dropout = nn.Dropout(p=self.dropout)

        # Attention
        if attn_dim is None:
            attn_dim = min(128, max(64, self.encoder_dim // 2))
        self.attention = AttentionBlock(self.encoder_dim, self.decoder_hidden, attn_dim)

        # Decoder RNN
        self.decoder = rnn_class(
            self.encoder_dim,                 # decoder consumes encoder vector (repeated per timestep)
            self.decoder_hidden,
            num_layers=self.decoder_num_layers,
            batch_first=True,
            dropout=self.dropout if self.decoder_num_layers > 1 else 0.0,
            bidirectional=False
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.decoder_hidden, max(self.decoder_hidden, self.input_size)),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(max(self.decoder_hidden, self.input_size), self.input_size)
        )

    def forward(self, x):
        """
        x: (B, T, input_size)
        returns: (B, input_size)
        """
        # Encoder
        enc_seq, _ = self.encoder(x)           # (B, T, encoder_dim)
        last_enc = enc_seq[:, -1, :]           # (B, encoder_dim)

        # Encoder FC block with residual
        z = self.enc_fc1(last_enc)             # (B, mid_dim)
        z = self.enc_bn1(z)
        z = self.enc_act(z)
        z = self.enc_dropout(z)

        z2 = self.enc_fc2(z)                   # (B, encoder_dim)
        z2 = self.enc_bn2(z2)
        z2 = self.enc_act(z2)
        z2 = self.enc_dropout(z2)

        enc_final = last_enc + z2              # (B, encoder_dim)  residual

        # Repeat encoder vector across time dimension for decoder input
        seq_len = x.size(1)
        dec_in = enc_final.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, encoder_dim)

        # Decoder
        dec_seq, _ = self.decoder(dec_in)      # (B, T, decoder_hidden)
        last_dec = dec_seq[:, -1, :]           # (B, decoder_hidden)

        # Attention over encoder sequence using decoder final state
        attn_ctx = self.attention(enc_seq, last_dec)  # (B, decoder_hidden)

        # Residual fusion
        dec_fused = last_dec + attn_ctx                # (B, decoder_hidden)

        # Output head
        out = self.output_head(dec_fused)              # (B, input_size)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        return {
            'input_size': self.input_size,
            'encoder_hidden': self.encoder_hidden,
            'decoder_hidden': self.decoder_hidden,
            'encoder_num_layers': self.encoder_num_layers,
            'decoder_num_layers': self.decoder_num_layers,
            'bidirectional_encoder': self.bidirectional_encoder,
            'dropout': self.dropout,
            'dropconnect_p': self.dropconnect_p,
            'use_gru': self.use_gru,
            'total_parameters': self.count_parameters()
        }


def create_model_from_config_new(config: dict, input_size: int):
    """
    Factory function with the same signature expected by the stored training script.
    Maps keys from the config dict to the new Encoder-Decoder architecture while
    preserving backwards compatibility.

    Expected keys in config (most already present in the search script):
      - bidirectional
      - use_gru
      - dropout
      - hidden_size (kept for compatibility)
      - num_layers
      - fc_* (ignored by architecture factory; FC head is internal)
    Optional keys (preferred to override defaults):
      - encoder_hidden
      - decoder_hidden
      - encoder_num_layers
      - decoder_num_layers
      - dropconnect
      - attention_heads
      - attn_dim
    """
    # Backwards-compatible defaults
    encoder_hidden = int(config.get('encoder_hidden', 256))
    decoder_hidden = int(config.get('decoder_hidden', 64))
    encoder_layers = int(config.get('encoder_num_layers', max(1, config.get('num_layers', 1))))
    decoder_layers = int(config.get('decoder_num_layers', 1))
    bidir = bool(config.get('bidirectional', False))
    use_gru = bool(config.get('use_gru', False))
    dropout = float(config.get('dropout', 0.2))
    dropconnect = float(config.get('dropconnect', config.get('dropconnect_p', 0.1)))
    attn_heads = int(config.get('attention_heads', 4))
    attn_dim = config.get('attn_dim', None)
    use_batch_norm = bool(config.get('use_batch_norm', True))

    model = EncoderDecoderLSTM(
        input_size=input_size,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        encoder_num_layers=encoder_layers,
        decoder_num_layers=decoder_layers,
        bidirectional_encoder=bidir,
        dropout=dropout,
        dropconnect_p=dropconnect,
        attn_heads=attn_heads,
        attn_dim=attn_dim,
        use_gru=use_gru,
        use_batch_norm=use_batch_norm
    )
    return model
    