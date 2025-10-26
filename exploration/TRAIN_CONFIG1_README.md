# Training Config 1 - Best Model from Brute Force Search

## What's Running

Training the best configuration discovered in the brute force search:
- **Config 1** achieved **Val R² = 0.3414** (best so far)
- This beat the previous best of 0.340

## Configuration Details

```python
{
    'lookback': 50,          # Long lookback window
    'hidden_size': 32,       # Small hidden size (efficient!)
    'num_layers': 1,         # Single LSTM layer
    'dropout': 0.3,
    'bidirectional': False,
    'use_gru': False,
    'fc_num_layers': 1,      # Direct linear output
    'fc_hidden_dims': 256,
    'fc_activation': 'relu',
    'fc_dropout': 0.0,
    'use_batch_norm': True,
    'batch_size': 32,
    'lr': 0.0001,
    'weight_decay': 0.0,
    'optimizer': 'adam',
    'grad_clip': 0.5,
    'lr_patience': 3
}
```

**Model Size:** Only 9,504 parameters (very efficient!)

## What the Script Does

`train_lstm_bruteforce.py` is a focused training script that:

1. ✅ Loads the predefined configuration
2. ✅ Trains the model on full dataset
3. ✅ Tracks training progress with early stopping
4. ✅ **Automatically creates submission folder** with:
   - `solution.py` - Competition-ready prediction code
   - `utils.py` - Required utilities
   - `model_best.pt` - Trained model weights
   - `metadata.json` - Training statistics

## Monitoring Progress

Check the log file:
```bash
tail -f wunderfund/exploration/logs/train_config1_*.log
```

The training will take approximately **1-1.5 hours** to complete 30 epochs.

## Expected Results

Based on the previous run:
- **Best epoch:** Around epoch 30
- **Val R²:** ~0.3414
- **Train R²:** ~0.3846
- **Overfitting gap:** ~0.0432 (acceptable)

## Output

Submission will be created at:
```
wunderfund/submissions/bruteforce_config1_r2_03414_YYYYMMDD_HHMMSS/
```

Ready to zip and submit!

## Usage Examples

### Train Config 1 (default):
```bash
python train_lstm_bruteforce.py --config config1 --device mps
```

### Train with custom epochs:
```bash
python train_lstm_bruteforce.py --config config1 --max-epochs 50
```

### Quick test on tiny dataset:
```bash
python train_lstm_bruteforce.py --config config1 --test
```

## Adding New Configurations

To add more predefined configs, edit the `CONFIGS` dictionary in `train_lstm_bruteforce.py`:

```python
CONFIGS = {
    'config1': { ... },
    'config2': {
        'lookback': 30,
        'hidden_size': 64,
        # ... your hyperparameters
    }
}
```

Then run:
```bash
python train_lstm_bruteforce.py --config config2
```

