# Wunder Challenge - Exploration & Baseline Submission

This directory contains exploratory analysis and baseline model development for the Wunder Challenge competition.

## 📊 Competition Summary

**Task:** Predict next market state vector (32 features) from sequence history
**Metric:** R² (coefficient of determination) averaged across all features
**Data:** 517 sequences × 1,000 steps each = 517,000 samples
**Features:** 32 anonymized numeric features per timestep

## 📁 Structure

```
exploration/
├── 00_create_sample_dataset.ipynb    # Creates sample data (50 sequences)
├── 01_exploratory_data_analysis.ipynb # Full EDA on training data
├── 02_baseline_solution_fast.ipynb   # Baseline models (fast version)
├── data/
│   ├── sample_full.parquet          # 50 sequences for dev
│   ├── sample_train.parquet         # 40 sequences for training
│   └── sample_val.parquet           # 10 sequences for validation
└── submissions/
    └── baseline_ema/
        ├── solution.py              # EMA model submission
        ├── utils.py                 # Required utilities
        └── baseline_ema.zip         # Submission package
```

## 🎯 Key Findings

### Data Characteristics
- **517 sequences**, each exactly **1,000 steps** long
- **32 features** named '0' through '31' (anonymized)
- **No missing values** in the dataset
- **138 MB** total dataset size
- **Sequences are independent** and randomly shuffled

### Baseline Performance

| Model | Train R² | Val R² | Description |
|-------|----------|---------|-------------|
| **EMA (α=0.1)** | **0.219** | **0.332** | Best baseline ⭐ |
| Moving Average | 0.174 | - | Mean of all history |
| Last Value | -0.427 | - | Naive persistence |

**Best Model:** Exponential Moving Average with α=0.1
- Simple, interpretable, and effective
- Good generalization (val > train suggests stable patterns)
- **Target to beat:** R² > 0.332

## 📦 Baseline Submission

### Files
- `submissions/baseline_ema/baseline_ema.zip` - Ready to submit
- Contains: `solution.py` + `utils.py`

### Model Details
**Exponential Moving Average (EMA)**
- Predicts next state as weighted average of past observations
- Formula: `EMA_t = α × state_t + (1-α) × EMA_{t-1}`
- Alpha = 0.1 (gives 90% weight to history, 10% to current)
- Maintains sequence state, resets between sequences

### Local Testing
```bash
cd submissions/baseline_ema
python3 -c "
from utils import ScorerStepByStep
from solution import PredictionModel

scorer = ScorerStepByStep('../../data/sample_val.parquet')
model = PredictionModel(alpha=0.1)
results = scorer.score(model)
print(f'Val R²: {results[\"mean_r2\"]:.6f}')
"
```

Expected output: `Val R²: 0.331577`

## 🚀 Next Steps

### Immediate Priorities
1. **LSTM/GRU Model Development** (Notebook 03)
   - Sequence modeling with recurrent networks
   - Target: R² > 0.4

2. **Training Pipeline** (Notebook 04)
   - Proper train/val/test splits
   - Hyperparameter tuning
   - Model checkpointing

3. **Advanced Architectures**
   - Transformer with attention
   - Mamba-2 for efficient sequence modeling
   - Ensemble methods

### Model Ideas
- **Feature Engineering:** Lagged features, rolling statistics, momentum indicators
- **Sequence Length:** Experiment with lookback windows (50, 100, 200 steps)
- **Multi-Task Learning:** Predict multiple steps ahead simultaneously
- **Attention Mechanisms:** Learn which past steps matter most

## 🔍 Usage

### Run Notebooks in Order
```bash
# 1. Create sample dataset (faster iteration)
jupyter notebook 00_create_sample_dataset.ipynb

# 2. Explore the data
jupyter notebook 01_exploratory_data_analysis.ipynb

# 3. Test baselines
jupyter notebook 02_baseline_solution_fast.ipynb
```

### Sample Data Stats
- **Train:** 40 sequences = 40,000 steps (~12 MB)
- **Val:** 10 sequences = 10,000 steps (~3 MB)
- **~10%** of full dataset, **much faster** for iteration

## 📝 Notes

- All notebooks tested and validated ✓
- Submission format verified ✓
- Uses sample data for speed (full data for final training)
- Baseline achieves R² = 0.332 (respectable start!)

## 🏆 Competition Strategy

1. **Phase 1 (Current):** Baseline established, infrastructure ready
2. **Phase 2:** LSTM/GRU models, beat R² > 0.4
3. **Phase 3:** Advanced architectures (Transformers, ensembles)
4. **Phase 4:** Full dataset training, final submission

**Current Best:** EMA α=0.1, R² = 0.332
**Next Target:** LSTM model, R² > 0.4
