# Brute Force Hyperparameter Search for LSTM Models

## Overview

This repository contains a comprehensive brute force hyperparameter search script (`exploration/tune_lstm_bruteforce.py`) designed for tuning LSTM/GRU models with automatic submission generation. The script performs exhaustive search over a predefined hyperparameter space, evaluates models on validation data, and automatically creates competition submissions when new best validation R² scores are achieved.

## Features

- **Exhaustive Hyperparameter Search**: Tests all combinations from a Cartesian product of hyperparameter values
- **Automatic Submission Generation**: Creates submission folders with trained models and solution code when validation R² improves
- **Full Data Utilization**: Uses 100% of available training data for maximum performance evaluation
- **Flexible Model Architecture**: Supports both LSTM and GRU models with configurable architecture components
- **Comprehensive Logging**: Dual logging to console and timestamped log files
- **Early Stopping**: Implements patience-based early stopping to prevent overfitting
- **Device Auto-Detection**: Automatically selects the best available device (CUDA/MPS/CPU)
- **Test Mode**: Includes test mode with tiny datasets for quick validation
- **Global Best Tracking**: Maintains and updates the best R² score across all configurations

## Architecture

### Core Components

1. **SequenceDataset Class**: Custom PyTorch Dataset for handling sequence data with configurable lookback windows
2. **Model Training Pipeline**:
   - `train_epoch()`: Single epoch training with gradient clipping and loss calculation
   - `validate()`: Model validation with R² score computation
   - `train_model()`: Complete training loop with early stopping and learning rate scheduling
3. **Hyperparameter Management**:
   - `sample_hyperparameters()`: Random sampling from search space
   - `build_search_space()`: Cartesian product generation for exhaustive search
4. **Submission System**:
   - `create_submission()`: Generates complete submission folders with model weights, solution code, and metadata
5. **Utility Functions**:
   - `setup_logging()`: Configures dual logging system
   - `detect_device()`: Hardware detection and selection
   - `calculate_r2()`: R² score computation across all features

### Model Architectures

The hyperparameter search optimizes multiple advanced neural network architectures for sequence prediction, implemented in `exploration/models.py`. Each architecture is designed to capture temporal dependencies in multi-feature time series data.

#### FlexibleRNN Architecture

The primary RNN-based model with extensive configuration options:

**RNN Backbone:**
- **Type**: LSTM or GRU (configurable via `use_gru`)
- **Directionality**: Unidirectional or Bidirectional (`bidirectional`)
- **Layers**: 1-4 stacked RNN layers (`num_layers`)
- **Hidden Size**: 64-256 units per layer (`hidden_size`)
- **Dropout**: 0.0-0.4 applied between layers (disabled for single-layer models)

**Sequence Processing:**
- **Lookback Window**: 20-100 timesteps for input sequence length
- **Feature Handling**: Multi-dimensional input processing with configurable feature dimensions
- **Output Extraction**: Uses final timestep output from RNN for prediction

**Output Head:**
- **Architecture**: Multi-layer fully connected network
- **Layers**: 1-3 fully connected layers (`fc_num_layers`)
- **Hidden Dimensions**: 64-512 units per hidden layer (`fc_hidden_dims`)
- **Activation Functions**: ReLU, Tanh, GELU, LeakyReLU (`fc_activation`)
- **Regularization**: Dropout (0.0-0.4) and optional Batch Normalization (`use_batch_norm`)

**Key Features:**
- Flexible output head with residual-style connections
- Configurable regularization to prevent overfitting
- Parameter counting for model complexity tracking
- Configuration serialization for reproducibility

#### RNNWithAttention Hybrid Architecture

Advanced hybrid model combining RNN temporal modeling with attention mechanisms:

**RNN Backbone:** (Same as FlexibleRNN)

**Attention Layer:**
- **Type**: Multi-Head Self-Attention over temporal dimension
- **Heads**: 4 attention heads (configurable)
- **Dropout**: Attention-specific dropout (0.1 default)
- **Integration**: Residual connections with layer normalization

**Processing Flow:**
1. RNN processes input sequence → temporal features
2. Multi-head attention captures feature interactions across timesteps
3. Layer normalization and residual connections stabilize training
4. Final timestep fed through configurable output head

**Advantages:**
- Captures long-range dependencies beyond RNN limitations
- Maintains temporal ordering while allowing feature interactions
- Improved performance on complex sequence patterns

#### EncoderDecoderLSTM Architecture

Advanced encoder-decoder architecture with attention and advanced regularization:

**Encoder:**
- **RNN Type**: Bidirectional LSTM/GRU (`bidirectional_encoder`)
- **Layers**: Configurable stacked layers (`encoder_num_layers`)
- **Hidden Size**: 256 units per direction (default)
- **Processing**: Sequence-to-vector encoding with final timestep extraction

**Encoder Enhancement Block:**
- **DropConnect Regularization**: Advanced weight dropout (0.1 default)
- **Residual Connections**: Maintains gradient flow
- **Batch Normalization**: Optional normalization layers
- **Multi-layer FC**: Projects encoder output through bottleneck

**Attention Mechanism:**
- **Type**: Additive attention between encoder sequence and decoder state
- **Projection Dimensions**: Configurable attention dimensionality
- **Context Integration**: Fuses attention context with decoder output

**Decoder:**
- **RNN Type**: Unidirectional LSTM/GRU
- **Input**: Repeated encoder vector across timesteps
- **Hidden Size**: 64 units (default)
- **Processing**: Vector-to-sequence decoding

**Output Head:**
- **Minimal Design**: Single FC layer with ReLU activation
- **Dropout**: Configurable regularization
- **Final Projection**: Maps to input feature dimensions

**Key Innovations:**
- Encoder-decoder separation for better sequence modeling
- DropConnect for improved generalization
- Attention-guided decoding for focused predictions
- Residual connections throughout the network

#### Architecture Factory Functions

**create_model_from_config(config, input_size):**
- Primary factory for FlexibleRNN and RNNWithAttention
- Maps hyperparameter config to appropriate model instantiation
- Supports attention flag for hybrid model selection

**create_model_from_config_new(config, input_size):**
- Advanced factory for EncoderDecoderLSTM
- Backwards-compatible with existing hyperparameter configs
- Maps legacy parameters to new architecture components

#### Training Configuration

**Optimization:**
- **Batch Size**: 64-256 samples per batch
- **Learning Rate**: 0.0001-0.001 with ReduceLROnPlateau scheduling
- **Weight Decay**: 0-0.01 for L2 regularization
- **Gradient Clipping**: None or 0.5-5.0 to prevent exploding gradients
- **Optimizer**: Adam or AdamW with configurable patience

**Regularization Techniques:**
- RNN dropout between layers
- FC layer dropout and batch normalization
- DropConnect in advanced architectures
- Early stopping with validation monitoring

**Performance Tracking:**
- R² score calculation across all features
- Overfitting gap monitoring (train - val R²)
- Parameter count tracking for model complexity
- Training time and convergence metrics

## Usage

### Prerequisites

- Python 3.7+
- PyTorch with CUDA/MPS support (optional)
- Required packages: numpy, pandas, scikit-learn, tqdm, torch

### Basic Usage

#### Test Mode (Recommended for initial testing)
```bash
python3 exploration/tune_lstm_bruteforce.py --test
```
Runs with tiny datasets and limited configurations for quick validation.

#### Full Production Run
```bash
python3 exploration/tune_lstm_bruteforce.py
```
Performs exhaustive search with 100 configurations and full datasets.

#### Custom Configuration
```bash
python3 exploration/tune_lstm_bruteforce.py --n_configs 50 --max_epochs 20 --device cuda
```
Allows customization of search parameters and hardware selection.

### Command Line Arguments

- `--test`: Enable test mode with tiny data (3 configs, 30 epochs max)
- `--n_configs`: Number of configurations to test (default: 100)
- `--max_epochs`: Maximum epochs per configuration (default: 100)
- `--patience`: Early stopping patience (default: 10)
- `--device`: Compute device (cpu/mps/cuda/auto, default: auto)
- `--best_r2`: Initial best R² threshold to beat (default: 0.34)
- `--shuffle`: Randomize configuration order (deterministic seed)
- `--max_configs`: Maximum configurations from Cartesian product (overrides --n_configs)

### Output Structure

The script generates several outputs:

- **Log Files**: `exploration/logs/bruteforce_TIMESTAMP.log`
- **Results CSV**: `exploration/models/bruteforce_search_results_TIMESTAMP.csv`
- **Submissions**: `submissions/bruteforce_r2_SCORE_TIMESTAMP_cfgID/`
- **Model Checkpoints**: Saved within submission folders
- **Metadata**: JSON files with configuration and performance details

### Submission Format

Each generated submission includes:
- `solution.py`: Competition-ready prediction code
- `model_best.pt`: PyTorch model checkpoint
- `utils.py`: Copied utility functions
- `metadata.json`: Configuration and performance metadata

## Data Requirements

- Training data in Parquet format (`competition_package/datasets/train.parquet`)
- Features: Multi-dimensional time series data
- Sequence indexing via `seq_ix` and `step_in_seq` columns
- Target features for prediction

## Performance Monitoring

The script provides real-time monitoring of:
- Per-epoch training/validation R² scores
- Overfitting gaps (train - val R²)
- Global best R² tracking
- Training time per configuration
- Model parameter counts

## Best Practices

1. **Start with Test Mode**: Always validate with `--test` before full runs
2. **Monitor Resources**: Large search spaces can be computationally intensive
3. **Review Logs**: Check log files for detailed training progress and errors
4. **Backup Results**: Important submissions are automatically saved
5. **Hardware Selection**: Use `--device cuda` for GPU acceleration when available

## Troubleshooting

- **Memory Issues**: Reduce batch size or hidden dimensions
- **Slow Training**: Consider GPU acceleration or smaller search spaces
- **No Improvements**: Adjust hyperparameter ranges or increase search space
- **Data Loading Errors**: Verify dataset paths and formats

## Dependencies

Core requirements:
```
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

Optional (for full functionality):
```
torchvision  # For CUDA support
torchaudio   # For MPS support
```

## Contributing

When modifying the search space or model architecture:
1. Update hyperparameter ranges in `build_search_space()`
2. Test changes in `--test` mode first
3. Validate submission generation
4. Update this README with any new features

## License

This project is part of the Wunderfund competition framework. See competition package for licensing details.
