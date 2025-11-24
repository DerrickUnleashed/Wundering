"""
Train LSTM model using pytorch-forecasting and create submission.

This script leverages the pytorch-forecasting library to train a model with specified
hyperparameters and automatically creates a submission folder compatible with the
competition setup.

Usage:
    python train_lstm_pytorch_forecasting.py --config config1  # Use predefined config
    python train_lstm_pytorch_forecasting.py --device cpu --max-epochs 40
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch.tuner import Tuner
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Predefined Configurations
# ============================================================================

CONFIGS = {
    'config1': {
        # Mapped from original config1 to pytorch-forecasting parameters
        'lookback': 50,              # max_encoder_length
        'horizon': 1,                # max_prediction_length
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'batch_size': 256,
        'lr': 0.0001,
        'weight_decay': 0.0,
        'optimizer': 'adam',
        'lr_patience': 3,
        'max_epochs': 40,
        'gradient_clip_val': 0.5,
        'use_batch_norm': True,      # not directly used in pytorch-forecasting LSTM
        'add_relative_time_idx': True,
        'add_target_scales': True,
        'add_encoder_length': True,
    }
}

# ============================================================================
# Helper Functions
# ============================================================================
def prepare_datasets(train_df, val_df, feature_cols, config):
    """Prepare pytorch-forecasting TimeSeriesDataSet for train and val."""

    # pytorch-forecasting expects a time_idx column for the sequence position
    # Our data has 'step_in_seq' that is suitable

    # We will treat 'seq_ix' as group id for multiple time series

    # Import dependencies
    from pytorch_forecasting.data import MultiNormalizer, TorchNormalizer
    import numpy as np

    max_encoder_length = config['lookback']
    max_prediction_length = config['horizon']

    # Combine train and validation data to ensure all categories are known
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # pytorch-forecasting requires MultiNormalizer for multiple targets
    # Using a list of (str, transformer) tuples, but fit needs list of data arrays
    normalizers = [TorchNormalizer(method="identity") for _ in feature_cols]
    normalizer = MultiNormalizer(normalizers)
    # Fit normalizer on training target data explicitly (list of 1D arrays)
    target_data = np.array([train_df[col].astype(np.float32).values for col in feature_cols]).T
    normalizer.fit(target_data)

    # Create dataset on combined data
    full_dataset = TimeSeriesDataSet(
        combined_df,
        time_idx="step_in_seq",
        target=feature_cols,
        group_ids=["seq_ix"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=feature_cols,
        add_relative_time_idx=config['add_relative_time_idx'],
        add_target_scales=config['add_target_scales'],
        add_encoder_length=config['add_encoder_length'],
        target_normalizer=normalizer,
        allow_missing_timesteps=True,
    )

    # Create validation set from the full dataset
    validation = TimeSeriesDataSet.from_dataset(full_dataset, val_df, stop_randomization=True)

    # Create training set from the full dataset
    training = TimeSeriesDataSet.from_dataset(full_dataset, train_df, stop_randomization=True)

    return training, validation


def get_dataloaders(training, validation, config):
    train_dataloader = training.to_dataloader(train=True, batch_size=config['batch_size'], num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=config['batch_size']*2, num_workers=0)
    return train_dataloader, val_dataloader


def build_model(training, config):
    """Build pytorch-forecasting TemporalFusionTransformer model configured for LSTM architecture."""

    # pytorch-forecasting does not provide an out-of-the-box pure LSTM model, but
    # TemporalFusionTransformer uses LSTM as backbone. We'll configure it with:
    # hidden_size, lstm_layers set to num_layers from config,
    # dropout to dropout config value,
    # attention to False (to approximate plain LSTM prediction),
    # learning_rate will be passed to optimizer in trainer.

    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=config["hidden_size"],
        lstm_layers=config["num_layers"],
        dropout=config["dropout"],
        attention_head_size=1,            # Set attention heads low to simulate less attention
        output_size=len(training.target_names),
        loss=torch.nn.MSELoss(),
        log_interval=10,
        reduce_on_plateau_patience=config["lr_patience"],
    )

    return model


def train_model(config, train_df, val_df, feature_cols, device):
    print("\n" + "=" * 70)
    print("TRAINING MODEL WITH PYTORCH-FORECASTING")
    print("=" * 70)
    print(f"Configuration: {config}")
    print(f"Device: {device}")
    print("=" * 70)

    training, validation = prepare_datasets(train_df, val_df, feature_cols, config)
    train_dataloader, val_dataloader = (
        training.to_dataloader(train=True, batch_size=config["batch_size"], num_workers=0),
        validation.to_dataloader(train=False, batch_size=config["batch_size"] * 2, num_workers=0),
    )

    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=config["hidden_size"],
        lstm_layers=config["num_layers"],
        dropout=config["dropout"],
        attention_head_size=1,
        loss=torch.nn.MSELoss(),
        output_size=[1 for _ in feature_cols],
        log_interval=10,
        reduce_on_plateau_patience=config["lr_patience"],
        learning_rate=config["lr"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best_model",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=config["lr_patience"], verbose=True)
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices=1,
        gradient_clip_val=config.get("gradient_clip_val", 0.0),
        callbacks=[checkpoint_callback, early_stop_callback, lr_logger],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path).to(device)

    best_model.eval()

    from sklearn.metrics import r2_score

    def evaluate(dataloader):
        actuals, predictions = [], []
        for x, y in dataloader:
            with torch.no_grad():
                y_pred = best_model(x)
            predictions.append(torch.stack(y_pred, dim=1).cpu())
            actuals.append(torch.stack(y[0], dim=1).cpu())
        preds = torch.cat(predictions).numpy().flatten()
        targets = torch.cat(actuals).numpy().flatten()
        return r2_score(targets, preds)
    train_r2 = evaluate(train_dataloader)
    val_r2 = evaluate(val_dataloader)
    overfitting_gap = train_r2 - val_r2

    print(f"\nBest Val R²: {val_r2:.6f}")
    print(f"Best Train R²: {train_r2:.6f}")
    print(f"Overfitting Gap: {overfitting_gap:.6f}")

    result = {
        "best_val_r2": val_r2,
        "best_train_r2": train_r2,
        "overfitting_gap": overfitting_gap,
        "best_epoch": checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
        "total_epochs": trainer.current_epoch + 1,
        "n_parameters": sum(p.numel() for p in best_model.parameters() if p.requires_grad),
        "best_model_path": best_model_path,
        "history": None,
    }
    return result, best_model, training


def create_submission(config, result, best_model, training, feature_cols, config_name, timestamp):
    """Create competition submission folder."""

    submission_name = f"pytorch_forecasting_{config_name}_r2_{int(result['best_val_r2']*1e4)}_{timestamp}"
    submission_dir = Path(f"../submissions/{submission_name}")
    submission_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("CREATING SUBMISSION")
    print("="*70)
    print(f"  Val R²: {result['best_val_r2']:.6f}")
    print(f"  Directory: {submission_dir}")

    # Save best model checkpoint to submission directory
    model_filename = "model_best.ckpt"
    model_path = submission_dir / model_filename
    shutil.copy(result['best_model_path'], model_path)
    print(f"  Saved model checkpoint as {model_filename}")

    # Generate solution.py adapted for pytorch-forecasting
    solution_code = f'''"""
Pytorch Forecasting LSTM Solution - {config_name}
Timestamp: {timestamp}

Best validation R²: {result['best_val_r2']:.6f}
Training R²: {result['best_train_r2']:.6f}
Overfitting gap: {result['overfitting_gap']:.6f}

Uses TemporalFusionTransformer with LSTM backbone from pytorch-forecasting.

Requirements:
- pytorch-lightning
- pytorch-forecasting
"""

import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from utils import DataPoint

class PredictionModel:
    def __init__(self, model_path="{model_filename}", lookback={config['lookback']}):
        self.device = torch.device("cpu")
        self.model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.lookback = lookback
        self.sequence_history = []
        self.current_seq_ix = None
        self.feature_cols = {feature_cols}

        # Prepare dataset prototype for feature info
        self.training = None  # Will be set externally for full context if needed

    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Reset if sequence changed
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        self.sequence_history.append(data_point.state.copy())

        if not data_point.need_prediction:
            return None

        if len(self.sequence_history) < self.lookback:
            # Fallback - mean of known states
            return np.mean(self.sequence_history, axis=0)

        # Prepare input batch for pytorch-forecasting model
        # Construct DataFrame with proper columns for a single batch
        import pandas as pd
        
        seq_len = self.lookback
        idxs = range(seq_len)
        data = {{
            "seq_ix": [self.current_seq_ix]*seq_len,
            "step_in_seq": list(idxs),
        }}
        for i, feature in enumerate(self.feature_cols):
            data[feature] = [st[i] for st in self.sequence_history[-seq_len:]]

        df = pd.DataFrame(data)
        # Create TimeSeriesDataSet for prediction without target (simulate encoder input)
        dataset = TimeSeriesDataSet(
            df,
            time_idx="step_in_seq",
            target=self.feature_cols,
            group_ids=["seq_ix"],
            max_encoder_length=self.lookback,
            max_prediction_length=1,
            time_varying_unknown_reals=self.feature_cols,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        dataloader = dataset.to_dataloader(batch_size=1, train=False)

        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                x = {{k: v.to(self.device) for k, v in x.items()}}
                prediction = self.model.predict(x)
                prediction = prediction.cpu().numpy()[0]
                return prediction
'''

    solution_path = submission_dir / 'solution.py'
    with open(solution_path, 'w') as f:
        f.write(solution_code)
    print(f"  Generated: solution.py")

    # Copy utils.py from competition package for compatibility
    utils_src = Path('../competition_package/utils.py')
    utils_dst = submission_dir / 'utils.py'
    shutil.copy(utils_src, utils_dst)
    print(f"  Copied: utils.py")

    # Save metadata
    metadata = {
        'config_name': config_name,
        'timestamp': timestamp,
        'val_r2': result['best_val_r2'],
        'train_r2': result['best_train_r2'],
        'overfitting_gap': result['overfitting_gap'],
        'best_epoch': result['best_epoch'],
        'total_epochs': result['total_epochs'],
        'n_parameters': result['n_parameters'],
        'config': config
    }
    metadata_path = submission_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: metadata.json")

    print("="*70)
    print(f"\n✅ Submission created: {submission_dir}")

    return submission_dir


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM model using pytorch-forecasting and create submission"
    )
    parser.add_argument(
        '--config',
        type=str,
        choices=list(CONFIGS.keys()),
        default='config1',
        help='Predefined configuration to use',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use for training',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=40,
        help='Maximum training epochs',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Use smaller test dataset',
    )

    args = parser.parse_args()

    # Setup device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("\n" + "=" * 70)
    print("LSTM TRAINING WITH PYTORCH-FORECASTING AND SUBMISSION CREATION")
    print("=" * 70)
    print(f"Configuration: {args.config}")

    print(f"Device: {device}")
    print(f"Max epochs: {args.max_epochs}")
    print("=" * 70)

    # Load configuration
    
    config = CONFIGS[args.config].copy()
    config['max_epochs'] = args.max_epochs

    # Load data
    print("\nLoading data...")
    if args.test:
        train_df = pd.read_parquet('data/tiny_train.parquet')
        val_df = pd.read_parquet('data/tiny_val.parquet')
    else:
        full_df = pd.read_parquet('../competition_package/datasets/train.parquet')
        all_seqs = full_df['seq_ix'].unique()

        # Split sequences 80/20
        np.random.seed(42)
        train_seqs = np.random.choice(all_seqs, size=int(0.8 * len(all_seqs)), replace=False)
        val_seqs = np.array([s for s in all_seqs if s not in train_seqs])

        train_df = full_df[full_df['seq_ix'].isin(train_seqs)].reset_index(drop=True)
        val_df = full_df[full_df['seq_ix'].isin(val_seqs)].reset_index(drop=True)

    feature_cols = [col for col in train_df.columns
                    if col not in ['seq_ix', 'step_in_seq', 'need_prediction']]

    print(f"Train: {len(train_df):,} rows, {train_df['seq_ix'].nunique()} sequences")
    print(f"Val:   {len(val_df):,} rows, {val_df['seq_ix'].nunique()} sequences")
    print(f"Features: {len(feature_cols)}")

    # Train model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_time = time.time()

    result, best_model, training = train_model(
        config, train_df, val_df, feature_cols, device
    )

    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time/60:.1f} minutes")

    # Create submission
    submission_dir = create_submission(
        config, result, best_model, training, feature_cols, args.config, timestamp
    )

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    print(f"\nSubmission ready at: {submission_dir}")
    print(f"\nTo create a zip file:")
    print(f"  cd ../submissions")
    print(f"  zip -r {submission_dir.name}.zip {submission_dir.name}/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
