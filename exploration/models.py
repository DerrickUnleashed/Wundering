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
